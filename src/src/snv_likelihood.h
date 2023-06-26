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
#include "coloring.h"

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

template <class Real_t> struct Genotype {
public:
  int altered;
  int cn;
  Genotype<Real_t>()=default;
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
  std::map<NodeHandle, std::map<size_t, Genotype<Real_t>>> node_to_snv_genotype;


  void calculate_inferred_genotypes(NodeHandle node,
                          SNVParams<Real_t> &p,
                          EventTree &tree,
                          size_t snv,
                          Attachment &a,
                          bool cn_overlap,
                          std::map<TreeLabel,
                          std::set<size_t>> &label_to_cell,
                          bool has_snv) {
    auto event = get_event_from_label(node->label);
    if (node != tree.get_root() && snvs[snv].overlaps_with_event(event)) {
      cn_overlap = true;
    }
    size_t cn = 0;
    Real_t result = 0.0;
    Genotype<Real_t> genotype(0,0);
    if (node_to_snv.find(node) != node_to_snv.end()) {
        has_snv = has_snv || node_to_snv[node].find(snv) != node_to_snv[node].end();
    }
    if (a.has_attached_cells(node->label) > 0 && has_snv) {
      for (auto cell : label_to_cell[node->label]) {
        cn = CN_matrix[cell][snv];
        break;
      }
      if (cn == 0) {
        ;
      } else if (cn_overlap) {
          Real_t max_score = 0.0;
          for (size_t al = 0; al <= cn; al++) {
            Real_t score = 0.0;
            auto prob = al == 0 ? p.e : (Real_t)al / (Real_t)cn;
            prob = std::min(prob, 1.0 - p.e);
            if (SNV_CLUSTERED) {
                Real_t D_sum = 0.0;
                Real_t B_sum = 0.0;
                for (auto cell : label_to_cell[node->label]) {
                    D_sum += D[cell][snv];
                    B_sum += B[cell][snv];
                }
                score = (D_sum - B_sum) * std::log(1.0 - prob) + B_sum * std::log(prob);
            } else {
                for (auto cell : label_to_cell[node->label]) {
                    score += (D[cell][snv] - B[cell][snv]) * std::log(1.0 - prob) + B[cell][snv] * std::log(prob);
                }
            }
            if (al == 0 || score > max_score) {
                genotype = Genotype<Real_t>(al, cn);
                max_score = score;
            }
         }
      } else {
        genotype = Genotype<Real_t>(1, cn);
      }


      if (node_to_snv_genotype.find(node) == node_to_snv_genotype.end()) {
        node_to_snv_genotype[node] = std::map<size_t, Genotype<Real_t>>();
      }
      node_to_snv_genotype[node][snv] = genotype;
    }

    for (auto child : tree.get_children(node)) {
      calculate_inferred_genotypes(child, p, tree, snv, a, cn_overlap, label_to_cell, has_snv);
    }
  }

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


  Real_t calculate_b_lik_for_SNV_acquire(
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
      result += calculate_b_lik_for_SNV_acquire(child, p, tree, snv, a,
                                                cn_overlap, label_to_cell);
    }
    return result;
  }










  Real_t calculate_b_lik_for_SNV_acquire_clustered(
      NodeHandle node, SNVParams<Real_t> &p, EventTree &tree, size_t snv,
      Attachment &a, bool cn_overlap,
      std::map<TreeLabel, std::set<size_t>> &label_to_cell) {
    auto event = get_event_from_label(node->label);
    if (node != tree.get_root() && snvs[snv].overlaps_with_event(event)) {
      cn_overlap = true;
    }
    Real_t result = 0.0;
    if (a.has_attached_cells(node->label) > 0) {
        Real_t D_sum = 0;
        size_t cn = 0;
        Real_t B_sum = 0.0;
        for (auto cell : label_to_cell[node->label]) {
        cn = CN_matrix[cell][snv];
        D_sum += D[cell][snv];
        B_sum += B[cell][snv];
        }
        if (cn != 0 && cn_overlap) {
          LogWeightAccumulator<Real_t> acc;
          for (size_t al = 0; al <= cn; al++) {
            auto prob = al == 0 ? p.e : (Real_t)al / (Real_t)cn;
            prob = std::min(prob, 1.0 - p.e);
            if (D_sum < B_sum) {
              throw "D smaller than B";
            }
             acc.add(-std::log(cn + 1) +
                      (D_sum - B_sum) * std::log(1.0 - prob) +
                      B_sum * std::log(prob));
          }
          result += acc.get_result();
          result -= (D_sum - B_sum) * std::log(1.0 - p.e) +
                    B_sum * std::log(p.e);
        } else if (cn != 0) {
          auto prob = 1.0 / (Real_t)cn;
          prob = std::min(prob, 1.0 - p.e);
          result += (D_sum - B_sum) * std::log(1.0 - prob) +
                    B_sum * std::log(prob);
          result -= (D_sum - B_sum) * std::log(1.0 - p.e) +
                    B_sum * std::log(p.e);
        }
    }
    for (auto child : tree.get_children(node)) {
      result += calculate_b_lik_for_SNV_acquire_clustered(child, p, tree, snv, a,
                                                cn_overlap, label_to_cell);
    }
    return result;
  }


  Real_t calculate_b_lik_for_SNV(
      SNVParams<Real_t> &p, EventTree &tree, size_t snv,
      std::map<TreeLabel, NodeHandle> &label_to_node,
      std::map<TreeLabel, std::set<size_t>> &label_to_cell, Attachment &a
      ) {
    std::map<TreeLabel, std::list<Genotype<Real_t>>> node_to_genotype;

    get_possible_genotypes(tree.get_root(), tree, snv, false, false,
                           node_to_genotype, label_to_cell);

    Real_t result = 0.0;
    for (size_t cell = 0; cell < CN_matrix.size(); cell++) {
      TreeLabel label = a.cell_to_tree_label[cell];
      LogWeightAccumulator<Real_t> acc;
      auto genotypes = node_to_genotype[label];
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
    if (SNV_CONSTANT == 0.0) {
      return;
    }
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

    if (SNV_CONSTANT == 0.0) {
      return;
    }

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

  SNVSolver<Real_t>(CONETInputData<Real_t> &cells)
      : cells{cells}, likelihood{cells} {
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
  if (SNV_CONSTANT == 0.0) {
      return 0.0;
    }
                           likelihood.init(tree, at);
                           return likelihood.get_D_likelihood(p, tree,at, 0, cells.snvs.size(), true);
//    return insert_snv_events(tree, at, p, false);
  }


Real_t insert_snv_events(EventTree &tree, Attachment &at, SNVParams<Real_t> p,
                           bool all) {
    if (SNV_CONSTANT == 0.0) {
      return 0.0;
    }

    likelihood.init(tree, at);
    auto label_to_cell = at.get_node_label_to_cells_map();

    log_debug("Initialized snv likelihood calculator");
    auto label_to_node = map_label_to_node(tree);
    auto nodes = tree.get_descendants(tree.get_root());
    for (auto n : nodes) {
       likelihood.node_to_snv[n] = std::set<size_t>();
    }
    likelihood.node_to_snv_genotype = std::map<NodeHandle, std::map<size_t, Genotype<Real_t>>>();
    for (size_t snv = 0; snv < cells.snvs.size(); snv++) {
      std::map<NodeHandle, Real_t> cache;
      if (cells.snvs[snv].candidate == 0 && !all) {
        continue;
      }

      for (auto n : nodes) {
        Real_t score = 0.0;
           if (SNV_CLUSTERED) {
            score = likelihood.calculate_b_lik_for_SNV_acquire_clustered(n, p, tree, snv, at, false,label_to_cell);
        } else {
            score = likelihood.calculate_b_lik_for_SNV_acquire(n, p, tree, snv, at, false,label_to_cell);
        }
        cache[n] = score;
    }


      MaxTreeColoring<Real_t> coloring;
      auto score_colored = coloring.get_max_colored_score(tree.get_root(), tree, cache);

      for (auto n : std::get<1>(score_colored)) {
        likelihood.node_to_snv[n].insert(snv);
      }
    }
    for (size_t snv = 0; snv < cells.snvs.size(); snv++) {
       likelihood.calculate_inferred_genotypes(tree.get_root(), p, tree, snv, at,false, label_to_cell, false);
    }
    return likelihood.get_total_likelihood(p, tree, at, 0, cells.snvs.size(),
                                           all);
  }














//  Real_t insert_snv_eventsold(EventTree &tree, Attachment &at, SNVParams<Real_t> p,
//                           bool all) {
//    if (SNV_CONSTANT == 0.0) {
//      return 0.0;
//    }
//
//    likelihood.init(tree, at);
//    auto label_to_cell = at.get_node_label_to_cells_map();
//
//    log_debug("Initialized snv likelihood calculator");
//    auto label_to_node = map_label_to_node(tree);
//    auto nodes = tree.get_descendants(tree.get_root());
//    for (auto n : nodes) {
//       likelihood.node_to_snv[n] = std::set<size_t>();
//    }
//    for (size_t snv = 0; snv < cells.snvs.size(); snv++) {
//      std::map<NodeHandle, Real_t> cache;
//      if (cells.snvs[snv].candidate == 0 && !all) {
//        continue;
//      }
//      Real_t lik_full = 0.0;
//      std::set<NodeHandle> nodes_{nodes.begin(), nodes.end()};
//      nodes_.erase(tree.get_root());
//      bool snv_added = false;
//      do {
//        snv_added = false;
//        auto lik_node = get_best_snv_location(tree, p, snv, nodes_, label_to_node, label_to_cell, at, cache);
//        if (std::get<1>(lik_node) != nullptr) {
//          lik_full += std::get<0>(lik_node);
//          log_debug("Found better location for snv ", snv);
//          snv_added = true;
//          likelihood.node_to_snv[std::get<1>(lik_node)].insert(snv);
//          auto node = std::get<1>(lik_node);
//          auto parent = tree.get_parent(node);
//          while (parent != tree.get_root()) {
//            nodes_.erase(parent);
//            parent = tree.get_parent(parent);
//          }
//          log_debug("Deleting descendants");
//          for (auto desc : tree.get_descendants(node)) {
//            nodes_.erase(desc);
//          }
//          log_debug("Deleted descendants");
//        } else {
//          log_debug("Did not manage to find better location for snv ", snv);
//        }
//      } while (snv_added);
//
//      MaxTreeColoring<Real_t> coloring;
//      auto score_colored = coloring.get_max_colored_score(tree.get_root(), tree, cache);
//
////      if (lik_full > std::get<0>(score_colored) + 0.00001) {
////        log(snv, " Uwaga Scores vs: ", lik_full, " ", std::get<0>(score_colored));
////      }
//      auto colored = std::get<1>(score_colored);
////      bool bad = false;
//      for (auto n : std::get<1>(score_colored)) {
//        if(likelihood.node_to_snv[n].find(snv) == likelihood.node_to_snv[n].end()) {
//            auto label = tree.get_node_label(n);
//            log("-----------------------------------------");
//            log("Colored has plus node ", label.first, " ", label.second, " ", cache[n]);
//            log(snv, " Scores vs: ", lik_full, " ", std::get<0>(score_colored));
//            log("Colored");
//            for (auto n2: colored) {
//                auto l = tree.get_node_label(n2);
//                log("(", l.first, " ", l.second, ")", cache[n2]);
//            }
//
//             log("Heavy");
//            for (auto n2: tree.get_descendants(tree.get_root())) {
//                if (likelihood.node_to_snv[n2].find(snv) != likelihood.node_to_snv[n2].end()) {
//                auto l = tree.get_node_label(n2);
//                log("(", l.first, " ", l.second, ")",cache[n2]);
//                }
//            }
//        }
//      }
//      for (auto n : tree.get_descendants(tree.get_root())) {
//            if (likelihood.node_to_snv[n].find(snv) != likelihood.node_to_snv[n].end()) {
//                if(std::find(colored.begin(), colored.end(), n) == colored.end()){
//                    auto label = tree.get_node_label(n);
//                    log("-----------------------------------------");
//                    log("Heavy has plus node ",label.first, " ", label.second, " ", cache[n]);
//                    log(snv, " Scores vs: ", lik_full, " ", std::get<0>(score_colored));
//                    log("Colored");
//            for (auto n2: colored) {
//                auto l = tree.get_node_label(n2);
//                log("(", l.first, " ", l.second, ")", cache[n2]);
//            }
//
//             log("Heavy");
//            for (auto n2: tree.get_descendants(tree.get_root())) {
//                if (likelihood.node_to_snv[n2].find(snv) != likelihood.node_to_snv[n2].end()) {
//                auto l = tree.get_node_label(n2);
//                log("(", l.first, " ", l.second, ")",cache[n2]);
//                }
//            }
//                }
//            }
//      }
//
//    }
//    return likelihood.get_total_likelihood(p, tree, at, 0, cells.snvs.size(),
//                                           all);
//  }
//
//  std::pair<Real_t, NodeHandle>
//  get_best_snv_location(EventTree &tree, SNVParams<Real_t> p, size_t snv,
//                        std::set<NodeHandle> &nodes,
//                        std::map<TreeLabel, NodeHandle> &label_to_node,
//                        std::map<TreeLabel, std::set<size_t>> &label_to_cell,
//                        Attachment &at, std::map<NodeHandle, Real_t> &cache) {
//    bool max_set = false;
//    NodeHandle max_node = nullptr;
//    Real_t max_lik = 0.0;
//    for (auto n : nodes) {
//      if (likelihood.node_to_snv.count(n) == 0) {
//        likelihood.node_to_snv[n] = std::set<size_t>();
//      }
//      Real_t lik = 0.0;
//       lik = score_node(tree, n, p, snv, nodes, label_to_node, label_to_cell, at, cache);
//      if (lik > max_lik) {
//        max_set = true;
//        max_lik = lik;
//        max_node = n;
//      }
//    }
//    return std::make_pair(max_lik, max_node);
//  }
//
//
//  Real_t score_node(EventTree &tree,
//                        NodeHandle node,
//                        SNVParams<Real_t> p, size_t snv,
//                        std::set<NodeHandle> &nodes,
//                        std::map<TreeLabel, NodeHandle> &label_to_node,
//                        std::map<TreeLabel, std::set<size_t>> &label_to_cell,
//                        Attachment &at,
//                        std::map<NodeHandle, Real_t> &cache) {
//
//    auto nodes_ = nodes;
//    Real_t score = 0.0;
//    if (cache.find(node) != cache.end()) {
//        score = cache[node];
//    } else if (SNV_CLUSTERED) {
//        score += likelihood.calculate_b_lik_for_SNV_acquire_clustered(node, p, tree, snv, at, false,label_to_cell);
//    } else {
//        score += likelihood.calculate_b_lik_for_SNV_acquire(node, p, tree, snv, at, false,label_to_cell);
//    }
//    cache[node] = score;
//
//    auto parent = tree.get_parent(node);
//    while (parent != tree.get_root()) {
//        nodes_.erase(parent);
//        parent = tree.get_parent(parent);
//    }
//    for (auto desc : tree.get_descendants(node)) {
//        nodes_.erase(desc);
//    }
//    bool snv_added = true;
//    while (snv_added) {
//        snv_added = false;
//        NodeHandle next_n = nullptr;
//        Real_t max_score = 0.0;
//        for (auto n : nodes_) {
//            Real_t node_score = score_node(tree,
//                        n,
//                        p,
//                        snv,
//                        nodes_,
//                        label_to_node,
//                        label_to_cell,
//                        at,
//                        cache);
//            if (next_n == nullptr || max_score < node_score) {
//                next_n = n;
//                max_score = node_score;
//            }
//         }
//        if (next_n != nullptr && max_score > 0) {
//            score += cache[next_n];
//            auto parent = tree.get_parent(next_n);
//            while (parent != tree.get_root()) {
//                nodes_.erase(parent);
//                parent = tree.get_parent(parent);
//            }
//            for (auto desc : tree.get_descendants(next_n)) {
//                nodes_.erase(desc);
//            }
//            snv_added = true;
//        }
//    }
//  return score;
//  }
};



#endif // !SNV_LIKELIHOOD_H