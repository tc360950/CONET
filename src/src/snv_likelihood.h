#ifndef SNV_LIKELIHOOD_H
#define SNV_LIKELIHOOD_H
#include <map>
#include <cmath>
#include <numeric> 
#include "tree/event_tree.h"
#include "tree/attachment.h"

#include <boost/math/distributions/negative_binomial.hpp>
#include <boost/math/distributions/binomial.hpp>

template < class Real_t> class CNMatrixCalculator {
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
  void move_cells_to_parent(EventTree::NodeHandle child, EventTree::NodeHandle parent, std::map<TreeLabel, std::set<size_t>> &attachment) {
    if (attachment.find(child->label) == attachment.end()) {
      return;
    }
    if (attachment.find(parent->label) == attachment.end()) {
      attachment[parent->label] = std::set<size_t>();
    }
    attachment[parent->label].insert(attachment[child->label].begin(), attachment[child->label].end());
    attachment.erase(child->label);
  }

  void calculate_CN_for_bins_at_node(EventTree::NodeHandle node, std::map<TreeLabel, std::set<size_t>> &attachment) {
    if (attachment.find(node->label) == attachment.end()) {
      return;
    }
    if (!is_cn_event(node->label)){
      return;
    }
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

    for (auto cell : attachment[node->label]) {
      for (size_t bin = event.first; bin < event.second; bin++) {
        if (CN_matrix[cell][bin] < 0) {
          cluster_to_counts_sum[event_clusters[bin]] += sum_counts[cell][bin];
          cluster_to_squared_counts_sum[event_clusters[bin]] += squared_counts[cell][bin];
          cluster_to_bin_count[event_clusters[bin]] += counts_score_length_of_bin[bin];
        }
      }
    }

    for (const auto &cluster : cluster_to_counts_sum) {
      if (cluster_to_bin_count[cluster.first] > 0.0) {
        auto bin_count = cluster_to_bin_count[cluster.first];
        cluster_to_CN[cluster.first] = std::round(cluster.second / bin_count);
      }
    }

    for (auto cell : attachment[node->label]) {
      for (size_t bin = event.first; bin < event.second; bin++) {
        if (CN_matrix[cell][bin] < 0) {
            CN_matrix[cell][bin] = cluster_to_CN[event_clusters[bin]];
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

  void _calculate_CN(EventTree::NodeHandle node, std::map<TreeLabel, std::set<size_t>> &attachment) {
    auto node_cache_id = cache_id;
    cache_id++;

    save_clustering_in_cache(event_clusters, node_cache_id);
    if (is_cn_event(node->label)) {
      update_clusters(event_clusters, get_event_from_label(node->label));      
    }

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
    std::map<TreeLabel, std::set<size_t>> attachment = at.get_node_label_to_cells_map();
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
      b.resize(cells.snvs.size());
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
using namespace boost::math;
using  ::boost::math::pdf;

template <class Real_t> class SNVLikelihood {
public:
    using NodeHandle = EventTree::NodeHandle;

    std::vector<std::vector<int>> CN_matrix; // cell, snv 
    std::vector<SNVEvent> snvs; 
    std::vector<int> cluster_sizes; 

    std::vector<std::vector<int>> B;
    std::vector<std::vector<int>> D;
    std::vector<std::vector<Real_t>> B_log_lik;
    std::vector<std::vector<Real_t>> D_log_lik;

    CNMatrixCalculator<Real_t> cn_calc;

    void calculate_d_lik(SNVParams<Real_t> &p) {
      for (size_t cell = 0; cell < D_log_lik.size(); cell++) {
        for (size_t snv = 0; snv < D_log_lik[cell].size(); snv++) {
            auto d = D[cell][snv];
            auto cn = CN_matrix[cell][snv];
            auto cluster_size = cluster_sizes[cell];

            Real_t mean = cn == 0 ? (cluster_size * p.e) : (cluster_size * p.m * cn);
            int num_failures = std::round(
              std::exp(
                std::log1p(-p.q) + std::log(mean) - std::log(p.q)
              )
            );
            D_log_lik[cell][snv] = 0 ? 0.0 : std::log(pdf(negative_binomial(num_failures, 1-p.q), d));
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

    void calculate_b_lik(SNVParams<Real_t> &p, EventTree &tree, Attachment &a) {
      auto label_to_node = map_label_to_node(tree);

      for (size_t cell = 0; cell < CN_matrix.size(); cell++) {
          //log_debug("Calculating likelihood for cell ", cell, " ", a.cell_to_tree_label.size());
          TreeLabel label = a.cell_to_tree_label[cell];
          //log_debug("Cell is attached to label: ", label_to_str(label));

          if (label_to_node.find(label) == label_to_node.end()) {
                      log_debug("Cell is attached to label: ", label_to_str(label));
            log_debug("UGHHH");
            for (auto x: label_to_node) {
              log_debug(label==x.first);

              log_debug(label_to_str(x.first), " ", x.second);
            }
          }
          auto genotypes = get_possible_genotypes(label_to_node[label], tree, cell);
          for (size_t snv = 0; snv < CN_matrix[cell].size(); snv++) {
              LogWeightAccumulator<Real_t> acc; 
              for (auto genotype: genotypes[snv]) {
                //log_debug("Processing genotype: ", genotype.altered, ", ", genotype.cn);
                //log_debug("D: ", D[cell][snv], " B: ", B[cell][snv]);
                 acc.add(
                    -std::log(genotypes.size())
                    + (
                      genotype.altered == 0 ?
                      std::log(pdf(binomial(D[cell][snv], p.e), B[cell][snv])) :
                      std::log(pdf(binomial(D[cell][snv], genotype.altered / genotype.cn), B[cell][snv])) // TODO tu na float? 
                    )
                 );
              }
          //log_debug("Updating likelihood for cell ", cell, " snv ", snv);
          B_log_lik[cell][snv] = acc.get_result();
          //log_debug("Result: ", acc.get_result());
        }
      }
    }

    std::vector<std::list<Genotype<Real_t>>> get_possible_genotypes(NodeHandle node, EventTree &tree, size_t cell) {
      //log_debug("Getting genotypes for node ", label_to_str(node->label) ," tree: ");
      //log_debug(TreeFormatter::to_string_representation(tree));
      std::list<NodeHandle> path_to_root;
      while (node != tree.get_root()) {
        path_to_root.push_back(node);
        node = node->parent;
      }
      path_to_root.reverse();
      //log_debug("Path to root is of size: ", path_to_root.size());
      std::vector<std::list<Genotype<Real_t>>> result; 
      result.reserve(CN_matrix[cell].size());
      std::vector<bool> snv_is_on_path;
      snv_is_on_path.resize(CN_matrix[cell].size());
      std::fill(snv_is_on_path.begin(), snv_is_on_path.end(), false);

      std::vector<bool> cn_change_after_alteration = snv_is_on_path;


      //log_debug("Traversing path to root...");
      for (auto node: path_to_root) {
        if (is_cn_event(node->label)) {
          auto event = get_event_from_label(node->label);
          for (size_t snv = 0; snv < CN_matrix[cell].size(); snv++) {
            if (snvs[snv].overlaps_with_event(event) && snv_is_on_path[snv]) {
              cn_change_after_alteration[snv] = true;
            } 
          }
        } else {
          snv_is_on_path[std::get<1>(node->label).snv] = true;
        }
      }
      for (size_t snv = 0; snv < CN_matrix[cell].size(); snv++) {
         std::list<Genotype<Real_t>> genotypes;
          if (!snv_is_on_path[snv] || CN_matrix[cell][snv] == 0) {
            genotypes.push_back(Genotype<Real_t>(0, CN_matrix[cell][snv]));
          } else if (!cn_change_after_alteration[snv]) {
            genotypes.push_back(Genotype<Real_t>(0, CN_matrix[cell][snv]));
          } else {
            for (size_t a = 0; a <= CN_matrix[cell][snv]; a++) {
              genotypes.push_back(Genotype<Real_t>(a, CN_matrix[cell][snv]));
            }
          } 
        result.push_back(genotypes);
      }
      //log_debug("Successfully claculated genotpyes.");
      return result;
    }

    Real_t _calculate_log_likelihood(SNVParams<Real_t> &p, EventTree &tree, Attachment &a) {
      log_debug("Calculating d likelihood....");
      calculate_d_lik(p);
      log_debug("Calculating b likelihood...");
      calculate_b_lik(p, tree, a);
      Real_t result = 0.0; 
      for (size_t cell = 0; cell < CN_matrix.size(); cell++) {
        result += std::accumulate(D_log_lik[cell].begin(), D_log_lik[cell].end(), 0.0);
        result += std::accumulate(B_log_lik[cell].begin(), B_log_lik[cell].end(), 0.0);
      }
      log_debug("Result: ", result);
      return result; 
    } 

    void fill_cn_matrix(std::vector<std::vector<int>> &bin_CN_matrix) {
      //log_debug("CN_matrix size ", CN_matrix.size(), ", ", CN_matrix[0].size());
      //log_debug("bin CN matrix size: ", bin_CN_matrix.size(), " ,", bin_CN_matrix[0].size());
      //log_debug("SNVs: ", snvs.size());
      for (size_t cell = 0; cell < CN_matrix.size(); cell++) {
        std::fill(CN_matrix[cell].begin(), CN_matrix[cell].end(), NEUTRAL_CN);
        for (size_t snv=0; snv < snvs.size(); snv++) {
          if (snvs[snv].lhs_locus != -1) {
            CN_matrix[cell][snv] = bin_CN_matrix[cell][snvs[snv].lhs_locus];
          }
        }
      }
    }

    Real_t calculate_log_likelihood(SNVParams<Real_t> p, EventTree &tree, Attachment &a) {
      log_debug("SNVLikelihood: calculating CN profiles...");
      cn_calc.calculate_CN(tree, a);
      log_debug("Filling CN profiles for SNVs...");
      fill_cn_matrix(cn_calc.CN_matrix);
      return _calculate_log_likelihood(p, tree, a);
    }

  SNVLikelihood<Real_t>(CONETInputData<Real_t> &cells): cn_calc(cells) {
    CN_matrix.resize(cells.get_cells_count());
    snvs = cells.snvs;
    cluster_sizes = cells.cluster_sizes; 
    B = cells.B;
    D = cells.D;
    B_log_lik.resize(cells.get_cells_count());
    D_log_lik.resize(cells.get_cells_count());

    for (size_t i = 0; i < cells.get_cells_count(); i++) {
        B_log_lik[i].resize(snvs.size());
        D_log_lik[i].resize(snvs.size());
    }
    for (auto &el : CN_matrix) {
      el.resize(snvs.size());
    }
  }
};
#endif // !SNV_LIKELIHOOD_H