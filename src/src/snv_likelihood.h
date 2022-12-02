#ifndef SNV_LIKELIHOOD_H
#define SNV_LIKELIHOOD_H
#include <map>
#include <cmath>
#include <set>
#include <numeric> 
#include <chrono>


#include "tree/event_tree.h"
#include "tree/attachment.h"

#include <boost/math/distributions/negative_binomial.hpp>
#include <boost/math/distributions/binomial.hpp>
template <class Real_t> class NegBinomCalc {

static NegBinomCalc<Real_t>* singleton;

public: 
  std::vector<Real_t> cache_;

  NegBinomCalc<Real_t> () {
    cache_.resize(10000000);
    cache_[0] = std::log(1);
    for (size_t i = 1; i < 10000000; i++) {
      cache_[i] = cache_[i-1] + std::log((Real_t) i);
    }
  }

  static NegBinomCalc<Real_t> *GetInstance();

   Real_t log_factorial(size_t n) {
    if (n < cache_.size()) {
      return cache_[n];
    }
    log("Cache miss ", n);
    Real_t result = cache_.back(); 
    for (size_t i = cache_.size(); i <= n; i++) {
      result += std::log( (Real_t) i);
    }
    return result;
  }
  Real_t get(size_t num_failures,  Real_t prob, size_t d) {
      Real_t result = num_failures * std::log(prob) + d * std::log(1.0 - prob);
      return result + log_factorial(num_failures + d - 1) - log_factorial(num_failures - 1) - log_factorial(d);
  }
};

template<class Real_t> NegBinomCalc<Real_t> * NegBinomCalc<Real_t>::singleton= nullptr;
/**
 * Static methods should be defined outside the class.
 */
template<class Real_t> NegBinomCalc<Real_t> *NegBinomCalc<Real_t>::GetInstance()
{
    /**
     * This is a safer way to create an instance. instance = new Singleton is
     * dangeruous in case two instance threads wants to access at the same time
     */
    if(singleton==nullptr){
        singleton = new NegBinomCalc<Real_t>();
    }
    return singleton;
}

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

    std::map<NodeHandle, std::set<size_t>> node_to_snv; 

    void calculate_d_lik(SNVParams<Real_t> &p, size_t start, size_t end) {
      for (size_t cell = 0; cell < D_log_lik.size(); cell++) {
        for (size_t snv = start; snv < end; snv++) {
            auto d = D[cell][snv];
            auto cn = CN_matrix[cell][snv];
            auto cluster_size = cluster_sizes[cell];

            Real_t mean = cn == 0 ? (cluster_size * p.e) : (cluster_size * p.m * cn);
            int num_failures = std::round(
              std::exp(
                std::log1p(-p.q) + std::log(mean) - std::log(p.q)
              )
            );
            
            //D_log_lik[cell][snv] = num_failures == 0 ? 0.0 : std::log(pdf(negative_binomial(num_failures, 1-p.q), d));
            D_log_lik[cell][snv] = num_failures == 0 ? 0.0 : NegBinomCalc<Real_t>::GetInstance()->get(num_failures, 1.0 - p.q, d);
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


    Real_t calculate_b_lik_for_SN_acquire2(NodeHandle node, SNVParams<Real_t> &p, EventTree &tree, size_t snv, 
   Attachment &a, bool cn_overlap, std::map<TreeLabel, std::set<size_t>> &label_to_cell) {
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
                  auto prob = al == 0 ? p.e : (Real_t) al / (Real_t) cn; 
                  prob = std::min(prob, 1.0 - p.e);
                  if (D[cell][snv] < B[cell][snv]) {
                    acc.add(
                      -std::log(cn+1)
                      - 10000.0
                    );
                  } else {
                    acc.add(
                      -std::log(cn+1) + (D[cell][snv] - B[cell][snv]) * std::log(1.0-prob) + B[cell][snv] * std::log(prob)
                    );
                  }
              }
              result += acc.get_result();
              result -= (D[cell][snv] - B[cell][snv]) * std::log(1.0-p.e) + B[cell][snv] * std::log(p.e);
            } else if (cn != 0) {
              auto prob = 1.0 / (Real_t) cn; 
              prob = std::min(prob, 1.0 - p.e);
              result += (D[cell][snv] - B[cell][snv]) * std::log(1.0-prob) + B[cell][snv] * std::log(prob);
              result -= (D[cell][snv] - B[cell][snv]) * std::log(1.0-p.e) + B[cell][snv] * std::log(p.e);
            }
          }
      }
      for (auto child : tree.get_children(node)) {
        result += calculate_b_lik_for_SN_acquire2(child, p, tree, snv, a, cn_overlap, label_to_cell);
      }
      return result; 
    }



    Real_t calculate_b_lik_for_SN_acquire(SNVParams<Real_t> &p, EventTree &tree, size_t snv, 
    std::map<TreeLabel, NodeHandle> &label_to_node, std::map<TreeLabel, std::set<size_t>> &label_to_cell, Attachment &a) {
      std::map<TreeLabel, std::list<Genotype<Real_t>>> node_to_genotype; 

      get_possible_genotypes(tree.get_root(), tree, snv, false ,false, node_to_genotype, label_to_cell);
 
      Real_t result = 0.0;
      for (size_t cell = 0; cell < CN_matrix.size(); cell++) {
          TreeLabel label = a.cell_to_tree_label[cell];
          LogWeightAccumulator<Real_t> acc; 
          auto genotypes = node_to_genotype[label];
          for (auto genotype: genotypes) {
            Real_t prob = genotype.altered == 0 ? p.e : (Real_t) genotype.altered / (Real_t) genotype.cn; 
            prob = std::min(prob, 1.0 - p.e);
            if (D[cell][snv] < B[cell][snv]) {
              acc.add(
                -std::log(genotypes.size())
                - 10000.0
              );
            } else {
              acc.add(
                -std::log(genotypes.size()) + (D[cell][snv] - B[cell][snv]) * std::log(1.0-prob) + B[cell][snv] * std::log(prob)
              );
            }
          }
          B_log_lik[cell][snv] = acc.get_result();
          result += acc.get_result();
      }
      return result;
    }

    void get_possible_genotypes(NodeHandle node, EventTree &tree, size_t snv, 
      bool snv_is_on_path, bool cn_change_after_alteration,
      std::map<TreeLabel, std::list<Genotype<Real_t>>> &result, std::map<TreeLabel, std::set<size_t>> &at) {
    
        auto event = get_event_from_label(node->label);
        if (node != tree.get_root() && std::find(node_to_snv[node].begin(), node_to_snv[node].end(), snv) != node_to_snv[node].end()) {
            snv_is_on_path=true;
        }
        if (node != tree.get_root() && snvs[snv].overlaps_with_event(event) && snv_is_on_path) {
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

        for (auto child: tree.get_children(node)) {
          get_possible_genotypes(child, tree, snv, snv_is_on_path, cn_change_after_alteration, result, at);
        }
    }

    void fill_cn_matrix(std::vector<std::vector<int>> &bin_CN_matrix) {
      for (size_t cell = 0; cell < CN_matrix.size(); cell++) {
        std::fill(CN_matrix[cell].begin(), CN_matrix[cell].end(), NEUTRAL_CN);
        for (size_t snv=0; snv < snvs.size(); snv++) {
          if (snvs[snv].lhs_locus != -1) {
            CN_matrix[cell][snv] = bin_CN_matrix[cell][snvs[snv].lhs_locus];
          }
        }
      }
    }

    Real_t get_total_likelihood(SNVParams<Real_t> p, EventTree &tree, Attachment &a, size_t start, size_t end) {
        end = std::min(end, snvs.size());
      auto label_to_node = map_label_to_node(tree);
      auto label_to_cell = a.get_node_label_to_cells_map();
      calculate_d_lik(p, start, end);
      log_debug("Calculating b likelihood...");
      for (size_t snv = start; snv < end; snv++) {
          calculate_b_lik_for_SN_acquire(p, tree, snv, label_to_node, label_to_cell, a);
      }
      Real_t result = 0.0; 
      for (size_t cell = 0; cell < CN_matrix.size(); cell++) {
        result += std::accumulate(D_log_lik[cell].begin()+ start, D_log_lik[cell].begin() + end, 0.0);
        result += std::accumulate(B_log_lik[cell].begin()+ start, B_log_lik[cell].begin() + end, 0.0);
      }
      log_debug("Result: ", result);
      return result; 
    }

  void init(EventTree& tree, Attachment &at) {
    log_debug("Calculating CN matrix for bins");
      cn_calc.calculate_CN(tree, at);
      log_debug("Calculating CN matrix for snvs");
      fill_cn_matrix(cn_calc.CN_matrix);
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



template <class Real_t> class SNVSolver {
public:
    using NodeHandle = EventTree::NodeHandle;
    CONETInputData<Real_t> &cells;
    SNVLikelihood<Real_t> likelihood; 

  SNVSolver<Real_t>(CONETInputData<Real_t> &cells): cells {cells}, likelihood {cells} {
  }


    std::map<TreeLabel, NodeHandle> map_label_to_node(EventTree &tree) {
        std::map<TreeLabel, NodeHandle> result;
        for (auto node : tree.get_descendants(tree.get_root())) {
          result[node->label] = node;
        }
        return result;
    }

  Real_t insert_snv_events(EventTree& tree, Attachment& at, SNVParams<Real_t> p) {
      if (SNV_CONSTANT == 0.0) {
        return 0.0;
      }
      likelihood.init(tree, at);

      log_debug("Initialized snv likelihood calculator");

      auto label_to_node = map_label_to_node(tree);
      auto label_to_cell = at.get_node_label_to_cells_map();

      for (size_t snv = 0; snv < cells.snvs.size(); snv++) {
        auto nodes = tree.get_descendants(tree.get_root());
        std::set<NodeHandle> nodes_{nodes.begin(), nodes.end()};
        nodes_.erase(tree.get_root());
        log_debug("Likelihood without snv for ", snv);
        bool snv_added = false;
        do {
          snv_added = false;
          auto lik_node = get_best_snv_location(tree, p, snv, nodes_, label_to_node, label_to_cell, at);
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
              for (auto desc: tree.get_descendants(std::get<1>(lik_node))) {
                nodes_.erase(desc);
              }
              log_debug("Deleted descendants");
          } else {
            log_debug("Did not manage to find better location for snv ", snv);
          }
        } while(snv_added);
      }
      return likelihood.get_total_likelihood(p, tree, at, 0, cells.snvs.size());
  }

  Real_t insert_snv_events(size_t move_count, EventTree& tree, Attachment& at, SNVParams<Real_t> p) {
      if (SNV_CONSTANT == 0.0) {
        return 0.0;
      }
      likelihood.init(tree, at);
      
      log_debug("Initialized snv likelihood calculator");

      size_t snvs_batch = 0;
      size_t batch_size = cells.snvs.size();

      if (SNV_BATCH_SIZE > 0) {
        size_t batches = cells.snvs.size() / SNV_BATCH_SIZE;
        if (batches * SNV_BATCH_SIZE < cells.snvs.size()) {
            batches++;
        }
        snvs_batch = (move_count % batches) * SNV_BATCH_SIZE;
        batch_size = SNV_BATCH_SIZE;
      }

      auto label_to_node = map_label_to_node(tree);
      auto label_to_cell = at.get_node_label_to_cells_map();

      for (size_t snv = snvs_batch; snv < snvs_batch + batch_size && snv < cells.snvs.size(); snv++) {
        auto nodes = tree.get_descendants(tree.get_root());
        std::set<NodeHandle> nodes_{nodes.begin(), nodes.end()};
        nodes_.erase(tree.get_root());
        log_debug("Likelihood without snv for ", snv);
        bool snv_added = false;
        do {
          snv_added = false;
          auto lik_node = get_best_snv_location(tree, p, snv, nodes_, label_to_node, label_to_cell, at);
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
              for (auto desc: tree.get_descendants(std::get<1>(lik_node))) {
                nodes_.erase(desc);
              }
              log_debug("Deleted descendants");
          } else {
            log_debug("Did not manage to find better location for snv ", snv);
          }
        } while(snv_added);
      }
      return likelihood.get_total_likelihood(p, tree, at, snvs_batch, snvs_batch + batch_size);
  }


  std::pair<Real_t, NodeHandle> get_best_snv_location(EventTree& tree, SNVParams<Real_t> p, size_t snv, 
  std::set<NodeHandle> &nodes,std::map<TreeLabel, NodeHandle> &label_to_node, std::map<TreeLabel, std::set<size_t>> &label_to_cell,
  Attachment &at) {
      bool max_set = false;
      NodeHandle max_node = nullptr; 
      Real_t max_lik = 0.0;

      for (auto n: nodes) {
          if (likelihood.node_to_snv.count(n) == 0) {
            likelihood.node_to_snv[n] = std::set<size_t>();
          }
          // likelihood.node_to_snv[n].insert(snv);
          auto lik = likelihood.calculate_b_lik_for_SN_acquire2(n, p, tree, snv, at, false, label_to_cell);
          // likelihood.node_to_snv[n].erase(snv);
          if (lik > max_lik) {
            max_set = true;
            max_lik = lik;
            max_node = n;
          }
      }
      return std::make_pair(max_lik, max_node);
  }
        

};


#endif // !SNV_LIKELIHOOD_H