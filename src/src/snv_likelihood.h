#ifndef SNV_LIKELIHOOD_H
#define SNV_LIKELIHOOD_H
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <fstream>
#include "tree/attachment.h"
#include "tree/event_tree.h"

template <class Real_t> class CNMatrixCalculator {
public:
  using NodeHandle = EventTree::NodeHandle;
  std::vector<std::vector<int>> CN_matrix; // cell, bin

  const size_t DEFAULT_CACHE_SIZE = 1000;

  std::vector<std::vector<Real_t>> sum_counts;
  std::vector<std::vector<Real_t>> squared_counts;
  std::vector<Real_t> counts_score_length_of_bin;

  // Will store current clustering induced by node
  std::vector<size_t> event_clusters;
  size_t max_cluster_id{0};
  // Used for persisting clusters induced by nodes
  std::vector<std::vector<size_t>> clusters_cache;
  size_t cache_id = 0;

  void save_clustering_in_cache(std::vector<size_t> &cluster, size_t cache_id) {
    if (cache_id >= clusters_cache.size()) {
      clusters_cache.resize(clusters_cache.size() + DEFAULT_CACHE_SIZE);
    }
    clusters_cache[cache_id] = cluster;
  }

  std::vector<size_t> get_clustering_from_cache(size_t cache_id) {
    return clusters_cache[cache_id];
  }

  /**
   * @brief Move all cells attached to @child to @parent
   */
  void move_cells_to_parent(EventTree::NodeHandle child,
                            EventTree::NodeHandle parent,
                            std::vector<TreeLabel> &attachment) {
    for (size_t c = 0; c < CN_matrix.size(); c++) {
      if (attachment[c] == child->label) {
        attachment[c] = parent->label;
      }
    }
  }

  void calculate_CN_for_bins_at_node(EventTree::NodeHandle node,
                                     std::vector<TreeLabel> &attachment) {
    std::map<size_t, Real_t> cluster_to_counts_sum;
    std::map<size_t, Real_t> cluster_to_bin_count;
    std::map<size_t, Real_t> cluster_to_squared_counts_sum;
    std::map<size_t, int> cluster_to_CN;

    auto event = get_event_from_label(node->label);

    for (size_t i = event.first; i < event.second; i++) {
      cluster_to_counts_sum[event_clusters[i]] = 0.0;
      cluster_to_bin_count[event_clusters[i]] = 0.0;
      cluster_to_squared_counts_sum[event_clusters[i]] = 0.0;
    }

    for (size_t c = 0; c < CN_matrix.size(); c++) {
      if (attachment[c] != node->label) {
        continue;
      }
      for (size_t bin = event.first; bin < event.second; bin++) {
        if (CN_matrix[c][bin] < 0) {
          cluster_to_counts_sum[event_clusters[bin]] += sum_counts[c][bin];
          cluster_to_squared_counts_sum[event_clusters[bin]] +=
              squared_counts[c][bin];
          cluster_to_bin_count[event_clusters[bin]] +=
              counts_score_length_of_bin[bin];
        }
      }
    }

    for (const auto &cluster : cluster_to_counts_sum) {
      if (cluster_to_bin_count[cluster.first] > 0.0) {
        auto bin_count = cluster_to_bin_count[cluster.first];
        cluster_to_CN[cluster.first] = std::round(cluster.second / bin_count);
      }
    }

    for (size_t c = 0; c < CN_matrix.size(); c++) {
      if (attachment[c] != node->label) {
        continue;
      }
      for (size_t bin = event.first; bin < event.second; bin++) {
        if (CN_matrix[c][bin] < 0) {
          CN_matrix[c][bin] = cluster_to_CN[event_clusters[bin]];
        }
      }
    }
  }

  void update_clusters(std::vector<size_t> &clusters, Event event) {
    max_cluster_id++;
    std::map<size_t, size_t> cluster_to_new_id;
    for (size_t i = event.first; i < event.second; i++) {
      if (cluster_to_new_id.find(clusters[i]) == cluster_to_new_id.end()) {
        cluster_to_new_id[clusters[i]] = max_cluster_id;
        max_cluster_id++;
      }
      clusters[i] = cluster_to_new_id[clusters[i]];
    }
  }

  void _calculate_CN(EventTree::NodeHandle node,
                     std::vector<TreeLabel> &attachment) {
    auto node_cache_id = cache_id;
    cache_id++;

    save_clustering_in_cache(event_clusters, node_cache_id);
    update_clusters(event_clusters, get_event_from_label(node->label));

    for (auto child : node->children) {
      _calculate_CN(child, attachment);
      move_cells_to_parent(child, node, attachment);
    }
    calculate_CN_for_bins_at_node(node, attachment);

    /* Restore clustering of parent */
    event_clusters = get_clustering_from_cache(node_cache_id);
  }

  void init_state() {
    if (clusters_cache.size() > DEFAULT_CACHE_SIZE) {
      clusters_cache.resize(DEFAULT_CACHE_SIZE);
    }
    for (auto &el : CN_matrix) {
      std::fill(el.begin(), el.end(), -1);
    }
    std::fill(event_clusters.begin(), event_clusters.end(), 0);
    max_cluster_id = 0;
    cache_id = 0;
  }

  void calculate_CN(EventTree &tree, Attachment &at) {
    init_state();
    std::vector<TreeLabel> attachment = at.cell_to_tree_label;
    for (auto node : tree.get_children(tree.get_root())) {
      _calculate_CN(node, attachment);
    }
    for (size_t cell = 0; cell < CN_matrix.size(); cell++) {
      for (auto &bin : CN_matrix[cell]) {
        if (bin < 0) {
          bin = NEUTRAL_CN;
        }
      }
    }
  }
  CNMatrixCalculator<Real_t>(CONETInputData<Real_t> &cells)
      : sum_counts{cells.get_summed_counts()},
        squared_counts{cells.get_squared_counts()},
        counts_score_length_of_bin{cells.get_counts_scores_regions()} {

    event_clusters.resize(cells.get_loci_count());

    CN_matrix.resize(cells.get_cells_count());
    for (auto &b : CN_matrix) {
      b.resize(cells.get_loci_count());
    }
    clusters_cache.resize(DEFAULT_CACHE_SIZE);
  }
};

template <class Real_t> class SNVParams {
public:
  Real_t e;
  Real_t m;
  Real_t q;

  SNVParams<Real_t>(Real_t e, Real_t m, Real_t q) {
    this->e = e;
    this->m = m;
    this->q = q;
  }
};

template <class Real_t> class Genotype {
public:
  int altered;
  int cn;
  Genotype<Real_t>(int a, int c) {
    cn = c;
    altered = a;
  }
};

template <class Real_t> class SNVLikelihood {
public:
  using NodeHandle = EventTree::NodeHandle;
  CONETInputData<Real_t> &cells;
  std::vector<std::vector<int>> CN_matrix; // cell, snv
  std::vector<SNVEvent> snvs;
  std::vector<int> cluster_sizes;

  std::vector<std::vector<int>> B;
  std::vector<std::vector<int>> D;
  std::vector<std::vector<Real_t>> Alpha;
  std::vector<std::vector<Real_t>> Beta;
  std::vector<std::vector<Real_t>> B_log_lik;
  std::vector<std::vector<Real_t>> D_log_lik;

  CNMatrixCalculator<Real_t> cn_calc;

  std::map<NodeHandle, std::set<size_t>> node_to_snv;

  void calculate_d_lik(SNVParams<Real_t> &p, size_t start, size_t end) {
    auto coef = std::exp(std::log(1 - p.q) - std::log(p.q));
    for (size_t cell = 0; cell < D_log_lik.size(); cell++) {
      for (size_t snv = start; snv < end; snv++) {
        Alpha[cell][snv] = cluster_sizes[cell] * p.e +
                           cluster_sizes[cell] * CN_matrix[cell][snv] * p.m;
        Beta[cell][snv] = 0.0;
        for (size_t i = 0; i < D[cell][snv]; i++) {
          Beta[cell][snv] += std::log(Alpha[cell][snv] * coef + i);
        }
        D_log_lik[cell][snv] = coef * std::log(1 - p.q) * Alpha[cell][snv] +
                               std::log(p.q) * D[cell][snv] + Beta[cell][snv];
      }
    }
  }

  std::map<TreeLabel, NodeHandle> map_label_to_node(EventTree &tree) {
    std::map<TreeLabel, NodeHandle> result;
    for (auto node : tree.get_descendants(tree.get_root())) {
      result[node->label] = node;
    }
    return result;
  }

  void calculate_b_lik_for_SN_acquire(
      SNVParams<Real_t> &p, EventTree &tree,
      Attachment &a,
      std::map<TreeLabel, std::set<size_t>> &label_to_cell,
      std::vector<std::vector<Real_t>> &result_buff_no_overlap,
      std::vector<std::vector<Real_t>> &result_buff_no_snv,
      std::vector<std::vector<Real_t>> &result_buff_overlap,
      bool all) {

    auto nodes =  tree.get_descendants(tree.get_root());
    for (size_t i = 0; i < nodes.size(); i++) {
        auto node = nodes[i];
        if (node == tree.get_root()) {
            continue;
        }
        for (size_t snv = 0; snv < cells.snvs.size(); snv++) {
              if (cells.snvs[snv].candidate == 0 && !all) {
                continue;
              }
              if (!a.has_attached_cells(node->label) > 0) {
                    continue;
              }
              result_buff_no_overlap[i][snv] = 0.0;
              result_buff_overlap[i][snv] = 0.0;
              result_buff_no_snv[i][snv] = 0.0;

              for (auto cell : label_to_cell[node->label]) {
                   auto cn = CN_matrix[cell][snv];
                   auto r = (D[cell][snv] - B[cell][snv]) * std::log(1.0 - p.e) + B[cell][snv] * std::log(p.e);
                   result_buff_no_snv[i][snv] += r;
                   if (cn == 0) {
                        result_buff_overlap[i][snv] += r;
                        result_buff_no_overlap[i][snv] += r;
                   } else {
                      auto prob = 1.0 / (Real_t)cn;
                      prob = std::min(prob, 1.0 - p.e);
                      result_buff_no_overlap[i][snv] += (D[cell][snv] - B[cell][snv]) * std::log(1.0 - prob) + B[cell][snv] * std::log(prob);

                      LogWeightAccumulator<Real_t> acc;
                      for (size_t al = 0; al <= cn; al++) {
                        prob = al == 0 ? p.e : (Real_t)al / (Real_t)cn;
                        prob = std::min(prob, 1.0 - p.e);
                        if (D[cell][snv] < B[cell][snv]) {
                          throw "D smaller than B";
                        }
                         acc.add(-std::log(cn + 1) +
                                  (D[cell][snv] - B[cell][snv]) * std::log(1.0 - prob) +
                                  B[cell][snv] * std::log(prob));
                      }
                      result_buff_overlap[i][snv] += acc.get_result();
                   }
              }
    }
  }
//  log("Ended");
  }

  Real_t calculate_b_lik_for_SN_acquire_at_node(
      std::map<NodeHandle, size_t> node_to_idx,
      NodeHandle node,
      SNVParams<Real_t> &p, EventTree &tree, size_t snv,
      Attachment &a,
      std::map<TreeLabel, std::set<size_t>> &label_to_cell,
      std::vector<std::vector<Real_t>> &result_buff_no_overlap,
      std::vector<std::vector<Real_t>> &result_buff_no_snv,
      std::vector<std::vector<Real_t>> &result_buff_overlap,
      bool cn_overlap) {

      Real_t result = 0.0;
      auto event = get_event_from_label(node->label);
      size_t idx = node_to_idx[node];
      if (node != tree.get_root() && snvs[snv].overlaps_with_event(event)) {
        cn_overlap = true;
      }
      if (cn_overlap) {
        result += result_buff_overlap[idx][snv] - result_buff_no_snv[idx][snv];
      } else {
        result += result_buff_no_overlap[idx][snv] - result_buff_no_snv[idx][snv];
      }
      for (auto child : tree.get_children(node)) {
            result += calculate_b_lik_for_SN_acquire_at_node(node_to_idx, child, p, tree, snv, a, label_to_cell, result_buff_no_overlap, result_buff_no_snv, result_buff_overlap, cn_overlap);
      }
      return result;
}






  Real_t calculate_b_lik_for_SN_acquire2(
      NodeHandle node, SNVParams<Real_t> &p, EventTree &tree, size_t snv,
      Attachment &a, bool cn_overlap,
      std::map<TreeLabel, std::set<size_t>> &label_to_cell) {
    auto event = get_event_from_label(node->label);
    if (node != tree.get_root() && snvs[snv].overlaps_with_event(event)) {
      cn_overlap = true;
    }
    Real_t result = 0.0;
    if (a.has_attached_cells(node->label) > 0) {
      for (auto cell : label_to_cell[node->label]) {
        auto cn = CN_matrix[cell][snv];
        if (cn != 0 && cn_overlap) {
          LogWeightAccumulator<Real_t> acc;
          for (size_t al = 0; al <= cn; al++) {
            auto prob = al == 0 ? p.e : (Real_t)al / (Real_t)cn;
            prob = std::min(prob, 1.0 - p.e);
            if (D[cell][snv] < B[cell][snv]) {
              throw "D smaller than B";
            }
             acc.add(-std::log(cn + 1) +
                      (D[cell][snv] - B[cell][snv]) * std::log(1.0 - prob) +
                      B[cell][snv] * std::log(prob));
          }
          result += acc.get_result();
          result -= (D[cell][snv] - B[cell][snv]) * std::log(1.0 - p.e) +
                    B[cell][snv] * std::log(p.e);
        } else if (cn != 0) {
          auto prob = 1.0 / (Real_t)cn;
          prob = std::min(prob, 1.0 - p.e);
          result += (D[cell][snv] - B[cell][snv]) * std::log(1.0 - prob) +
                    B[cell][snv] * std::log(prob);
          result -= (D[cell][snv] - B[cell][snv]) * std::log(1.0 - p.e) +
                    B[cell][snv] * std::log(p.e);
        }
      }
    }
    for (auto child : tree.get_children(node)) {
      result += calculate_b_lik_for_SN_acquire2(child, p, tree, snv, a,
                                                cn_overlap, label_to_cell);
    }
    return result;
  }

  Real_t calculate_b_lik_for_SNV(
      SNVParams<Real_t> &p, EventTree &tree, size_t snv,
      std::map<TreeLabel, NodeHandle> &label_to_node,
      std::map<TreeLabel, std::set<size_t>> &label_to_cell, Attachment &a
//      ,std::ofstream &f, bool save
      ) {
    std::map<TreeLabel, std::list<Genotype<Real_t>>> node_to_genotype;

    get_possible_genotypes(tree.get_root(), tree, snv, false, false,
                           node_to_genotype, label_to_cell);

    Real_t result = 0.0;
    for (size_t cell = 0; cell < CN_matrix.size(); cell++) {
      TreeLabel label = a.cell_to_tree_label[cell];
      LogWeightAccumulator<Real_t> acc;
      auto genotypes = node_to_genotype[label];

//      if (save && genotypes.size() > 1) {
//        f << cell << ";" << snv << ";" << genotypes.size() << ";" << genotypes.front().cn << ";"<<  label.first << ";" << label.second << ";"<<snvs[snv].overlaps_with_event(label) << ";" << snvs[snv].lhs_locus <<";FULL\n";
//      } else if (save && genotypes.front().altered == 0) {
//        f << cell << ";" << snv << ";" << genotypes.size() << ";" << genotypes.front().cn << ";"<<  label.first << ";" << label.second << ";"<<snvs[snv].overlaps_with_event(label)<< ";" << snvs[snv].lhs_locus << ";ZERO\n";
//      } else if (save && genotypes.front().altered == 1) {
//        f << cell << ";" << snv << ";" << genotypes.size() << ";" << genotypes.front().cn<< ";" <<  label.first << ";" << label.second << ";"<<snvs[snv].overlaps_with_event(label) << ";" << snvs[snv].lhs_locus << ";ONE\n";
//      } else if (save) {
//        f << cell << ";" << snv << ";" << genotypes.size() << ";" <<genotypes.front().cn << ";"<<  label.first << ";" << label.second << ";"<<snvs[snv].overlaps_with_event(label) << ";" << snvs[snv].lhs_locus <<";UNKNOWN\n";
//      }

      for (auto genotype : genotypes) {
        Real_t prob = genotype.altered == 0 || genotype.cn == 0
                          ? p.e
                          : (Real_t)genotype.altered / (Real_t)genotype.cn;
        prob = std::min(prob, 1.0 - p.e);
        acc.add(-std::log(genotypes.size()) +
              (D[cell][snv] - B[cell][snv]) * std::log(1.0 - prob) +
              B[cell][snv] * std::log(prob));
      }
      B_log_lik[cell][snv] = acc.get_result();
      result += acc.get_result();
    }
    return result;
  }

  void get_possible_genotypes(
      NodeHandle node, EventTree &tree, size_t snv, bool snv_is_on_path,
      bool cn_change_after_alteration,
      std::map<TreeLabel, std::list<Genotype<Real_t>>> &result,
      std::map<TreeLabel, std::set<size_t>> &at) {

    auto event = get_event_from_label(node->label);
    if (node != tree.get_root() &&
        std::find(node_to_snv[node].begin(), node_to_snv[node].end(), snv) !=
            node_to_snv[node].end()) {
      snv_is_on_path = true;
//      std::cout << "SNV is on path for " << node->label.first << ";" << node->label.second << " " << snv << "\n";
    }
    if (node != tree.get_root() && snvs[snv].overlaps_with_event(event) &&
        snv_is_on_path) {
      cn_change_after_alteration = true;
    }

    if (at.count(node->label) > 0) {
      auto cell = *(at[node->label].begin());
      std::list<Genotype<Real_t>> genotypes;
      if (!snv_is_on_path || CN_matrix[cell][snv] == 0) {
        genotypes.push_back(Genotype<Real_t>(0, CN_matrix[cell][snv]));
      } else if (!cn_change_after_alteration) {
        genotypes.push_back(Genotype<Real_t>(1, CN_matrix[cell][snv]));
      } else {
        for (size_t a = 0; a <= CN_matrix[cell][snv]; a++) {
          genotypes.push_back(Genotype<Real_t>(a, CN_matrix[cell][snv]));
        }
      }
      result[node->label] = genotypes;
    }

    for (auto child : tree.get_children(node)) {
      get_possible_genotypes(child, tree, snv, snv_is_on_path,
                             cn_change_after_alteration, result, at);
    }
  }

  void fill_cn_matrix(std::vector<std::vector<int>> &bin_CN_matrix) {
    for (size_t cell = 0; cell < CN_matrix.size(); cell++) {
      std::fill(CN_matrix[cell].begin(), CN_matrix[cell].end(), NEUTRAL_CN);
      for (size_t snv = 0; snv < snvs.size(); snv++) {
        if (snvs[snv].lhs_locus != -1) {
          CN_matrix[cell][snv] = bin_CN_matrix[cell][snvs[snv].lhs_locus];
        }
      }
    }
  }

  Real_t get_D_likelihood(SNVParams<Real_t> p, EventTree &tree,
                              Attachment &a, size_t start, size_t end,
                              bool all) {
    end = std::min(end, snvs.size());
    calculate_d_lik(p, start, end);
    Real_t result = 0.0;
    for (size_t cell = 0; cell < CN_matrix.size(); cell++) {
      for (size_t snv = start; snv < end; snv++) {
        if (cells.snvs[snv].candidate == 0 && !all) {
          continue;
        }
        result += D_log_lik[cell][snv];
      }
    }
    log_debug("Result: ", result);
    return result;
  }

  Real_t get_B_likelihood(SNVParams<Real_t> p, EventTree &tree,
                              Attachment &a, size_t start, size_t end,
                              bool all) {
//    std::ofstream f{"/data/genotypes.conet"};
//    std::ofstream f2{"/data/snvs.conet"};
//    for (auto node: tree.get_descendants(tree.get_root())) {
//        for (auto snv : node_to_snv[node]) {
//            f2 << node->label.first << ";" << node->label.second << ";" << snv << "\n";
//        }
//    }
    end = std::min(end, snvs.size());
    auto label_to_node = map_label_to_node(tree);
    auto label_to_cell = a.get_node_label_to_cells_map();
    log_debug("Calculating b likelihood...");
    for (size_t snv = start; snv < end; snv++) {
      if (cells.snvs[snv].candidate == 0 && !all) {
        continue;
      }
      calculate_b_lik_for_SNV(p, tree, snv, label_to_node, label_to_cell, a);
    }
    Real_t result = 0.0;
    for (size_t cell = 0; cell < CN_matrix.size(); cell++) {
      for (size_t snv = start; snv < end; snv++) {
        if (cells.snvs[snv].candidate == 0 && !all) {
          continue;
        }
        result += B_log_lik[cell][snv];
      }
    }
    log_debug("Result: ", result);
    return result;
  }

  Real_t get_total_likelihood(SNVParams<Real_t> p, EventTree &tree,
                              Attachment &a, size_t start, size_t end,
                              bool all) {
    return get_B_likelihood(p, tree, a, start, end, all) + get_D_likelihood(p, tree,a ,start, end, all);
  }

  void init(EventTree &tree, Attachment &at) {
    log_debug("Calculating CN matrix for bins");
    cn_calc.calculate_CN(tree, at);
    log_debug("Calculating CN matrix for snvs");
    fill_cn_matrix(cn_calc.CN_matrix);
    log_debug("B size", B.size(), " ", B[0].size());
    log_debug("D size", D.size(), " ", D[0].size());
    log_debug("CN size", CN_matrix.size(), " ", CN_matrix[0].size());
  }

  SNVLikelihood<Real_t>(CONETInputData<Real_t> &cells)
      : cells{cells}, cn_calc(cells) {
    CN_matrix.resize(cells.get_cells_count());
    snvs = cells.snvs;
    cluster_sizes = cells.cluster_sizes;
    B = cells.B;
    D = cells.D;
    B_log_lik.resize(cells.get_cells_count());
    D_log_lik.resize(cells.get_cells_count());
    Alpha.resize(cells.get_cells_count());
    Beta.resize(cells.get_cells_count());

    for (size_t i = 0; i < cells.get_cells_count(); i++) {
      B_log_lik[i].resize(snvs.size());
      D_log_lik[i].resize(snvs.size());
      Alpha[i].resize(snvs.size());
      Beta[i].resize(snvs.size());
      CN_matrix[i].resize(snvs.size());
    }
  }
};

template <class Real_t> class SNVSolver {
public:
  using NodeHandle = EventTree::NodeHandle;
  CONETInputData<Real_t> &cells;
  SNVLikelihood<Real_t> likelihood;
  std::vector<std::vector<Real_t>> result_buff_no_overlap;
  std::vector<std::vector<Real_t>> result_buff_no_snv;
  std::vector<std::vector<Real_t>> result_buff_overlap;

  SNVSolver<Real_t>(CONETInputData<Real_t> &cells)
      : cells{cells}, likelihood{cells} {

        result_buff_overlap.resize(1000);
        result_buff_no_overlap.resize(1000);
        result_buff_no_snv.resize(1000);
        for (size_t i = 0; i< 1000; i++) {
            result_buff_overlap[i].resize(cells.snvs.size());
            result_buff_no_overlap[i].resize(cells.snvs.size());
            result_buff_no_snv[i].resize(cells.snvs.size());
        }
      }

  std::map<TreeLabel, NodeHandle> map_label_to_node(EventTree &tree) {
    std::map<TreeLabel, NodeHandle> result;
    for (auto node : tree.get_descendants(tree.get_root())) {
      result[node->label] = node;
    }
    return result;
  }

  Real_t insert_snv_events(EventTree &tree, Attachment &at,
                           SNVParams<Real_t> p) {
    return insert_snv_events(tree, at, p, false);
  }

  Real_t insert_snv_events(EventTree &tree, Attachment &at, SNVParams<Real_t> p,
                           bool all) {
    if (SNV_CONSTANT == 0.0) {
      return 0.0;
    }

//    log("Intiialized");

    likelihood.init(tree, at);
    auto label_to_cell = at.get_node_label_to_cells_map();
    likelihood.calculate_b_lik_for_SN_acquire(
        p, tree, at, label_to_cell, result_buff_no_overlap, result_buff_no_snv, result_buff_overlap, all
    );

    log_debug("Initialized snv likelihood calculator");
    auto label_to_node = map_label_to_node(tree);
    auto nodes = tree.get_descendants(tree.get_root());
    std::map<NodeHandle, size_t> node_to_idx;
    for (size_t i = 0; i < nodes.size(); i++) {
        node_to_idx[nodes[i]] = i;
    }
    for (auto n : nodes) {
       likelihood.node_to_snv[n] = std::set<size_t>();
    }
    for (size_t snv = 0; snv < cells.snvs.size(); snv++) {
      if (cells.snvs[snv].candidate == 0 && !all) {
        continue;
      }
      std::set<NodeHandle> nodes_{nodes.begin(), nodes.end()};
      nodes_.erase(tree.get_root());
      bool snv_added = false;
      do {
        snv_added = false;
        auto lik_node = get_best_snv_location(node_to_idx, tree, p, snv, nodes_,
                                              label_to_node, label_to_cell, at);
        if (std::get<1>(lik_node) != nullptr) {
          log_debug("Found better location for snv ", snv);
          snv_added = true;
          likelihood.node_to_snv[std::get<1>(lik_node)].insert(snv);
          auto node = std::get<1>(lik_node);
          auto parent = tree.get_parent(node);
          while (parent != tree.get_root()) {
            nodes_.erase(parent);
            parent = tree.get_parent(parent);
          }
          log_debug("Deleting descendants");
          for (auto desc : tree.get_descendants(node)) {
            nodes_.erase(desc);
          }
          log_debug("Deleted descendants");
        } else {
          log_debug("Did not manage to find better location for snv ", snv);
        }
      } while (snv_added);
    }
    return likelihood.get_total_likelihood(p, tree, at, 0, cells.snvs.size(),
                                           all);
  }

  std::pair<Real_t, NodeHandle>
  get_best_snv_location(std::map<NodeHandle, size_t> node_to_idx, EventTree &tree, SNVParams<Real_t> p, size_t snv,
                        std::set<NodeHandle> &nodes,
                        std::map<TreeLabel, NodeHandle> &label_to_node,
                        std::map<TreeLabel, std::set<size_t>> &label_to_cell,
                        Attachment &at) {
    bool max_set = false;
    NodeHandle max_node = nullptr;
    Real_t max_lik = 0.0;

    for (auto n : nodes) {
      if (likelihood.node_to_snv.count(n) == 0) {
        likelihood.node_to_snv[n] = std::set<size_t>();
      }
      auto lik = likelihood.calculate_b_lik_for_SN_acquire_at_node(
        node_to_idx,
        n, p, tree, snv, at, label_to_cell,
      result_buff_no_overlap,
      result_buff_no_snv,
      result_buff_overlap, false);

      auto lik2 = likelihood.calculate_b_lik_for_SN_acquire2(n, p, tree, snv, at, false,label_to_cell);

//      auto lik_without = likelihood.get_total_likelihood(p, tree, at, snv, snv + 1, true);
//      likelihood.node_to_snv[n].insert(snv);
//      auto lik_with = likelihood.get_total_likelihood(p, tree, at, snv, snv + 1, true);
//      likelihood.node_to_snv[n].erase(snv);
//
//       if (std::abs(lik_with - lik_without - lik2) >= 0.001) {
//            std::cout << snv << " " << cells.snvs[snv].candidate << " " <<  lik << " versus " << lik_with - lik_without << " versus " << lik2 << "\n";
//       }
      if (lik2 > max_lik) {
        max_set = true;
        max_lik = lik;
        max_node = n;
      }
    }
    return std::make_pair(max_lik, max_node);
  }
};





//TODO sprawdzi  czyszczenie B i D likelihoodu!

#endif // !SNV_LIKELIHOOD_H