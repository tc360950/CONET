#ifndef VERTEX_SAMPLER_H
#define VERTEX_SAMPLER_H
#include <algorithm>
#include <cassert>
#include <vector>

#include "../utils/logger/logger.h"
#include "../utils/random.h"
#include "./utils/event_container.h"
#include "../types.h"

/**
 * This class is responsible for sampling labels for new tree vertices
 */
template <class Real_t> class VertexLabelSampler {
private:
  Real_t CN_LABEL_WEIGHT = std::log(0.5);
  Locus max_loci;
  EventContainer unused_labels;
  std::vector<Locus> chromosome_end_markers;
  std::vector<SNVEvent> unused_snvs; 
  size_t get_locus_chromosome(Locus locus) {
    size_t chromosome = 0;
    while (locus >= chromosome_end_markers[chromosome]) {
      chromosome++;
    }
    return chromosome;
  }

  bool is_valid_cn_label(TreeLabel label) {
      assert(is_cn_event(label));
      auto event = get_event_from_label(label);
      return is_valid_event(event) && get_locus_chromosome(event.first) ==
                                       get_locus_chromosome(event.second);
  }

  void init() {
    for (size_t brkp = 0; brkp <= max_loci; brkp++) {
      for (size_t brkp2 = brkp + 1; brkp2 <= max_loci; brkp2++) {
        if (is_valid_cn_label(TreeLabel(std::make_pair(brkp, brkp2)))) {
          unused_labels.insert(std::make_pair(brkp, brkp2));
        }
      }
    }
  }

public:
  VertexLabelSampler(size_t max_loci, std::vector<size_t> chr_markers, std::vector<SNVEvent> snvs)
      : max_loci{max_loci}, unused_labels{max_loci}, chromosome_end_markers{
                                                         chr_markers} {
    unused_snvs = snvs;
    init();
  }

  void add_cn_label(TreeLabel l) { assert(l.index() == 0); unused_labels.erase(get_event_from_label(l)); }

  void remove_cn_label(TreeLabel l) { assert(l.index() == 0);unused_labels.insert(get_event_from_label(l)); }

  Real_t get_sample_cn_label_log_kernel() {
    return -std::log((Real_t)unused_labels.size());
  }

  TreeLabel sample_cn_label(Random<Real_t> &random) {
    return TreeLabel(unused_labels.get_nth(random.next_int(unused_labels.size())));
  }

  bool has_free_cn_labels() { return !unused_labels.empty(); }


  void add_snv_label(TreeLabel l) { 
    assert(!is_cn_event(l));
    for (size_t i = 0; i < unused_snvs.size(); i++) {
      if (unused_snvs[i] == std::get<1>(l)) {
        unused_snvs.erase(unused_snvs.begin() + i);
        break;
      }
    }
  }

  void remove_snv_label(TreeLabel l) { 
    assert(!is_cn_event(l));
    unused_snvs.push_back(std::get<1>(l)); 
  }

  Real_t get_sample_snv_label_log_kernel() {
    return -std::log((Real_t)unused_snvs.size());
  }

  TreeLabel sample_snv_label(Random<Real_t> &random) {
    return TreeLabel(unused_snvs[random.next_int(unused_snvs.size())]);
  }

  bool has_free_snv_labels() { return !unused_snvs.empty(); }


  void add_label(TreeLabel l) { 
    if (is_cn_event(l)) {
      add_cn_label(l);
    } else {
      add_snv_label(l);
    }
  }

  void remove_label(TreeLabel l) { 
    if (is_cn_event(l)) {
      remove_cn_label(l);
    } else {
      remove_snv_label(l);
    }    
  }

  Real_t get_sample_label_log_kernel() {
    log_debug("VertexLabelSampler: Calculating sample label kernel...\n", unused_labels.size(), " ", unused_snvs.size());
    if (unused_labels.size() > 0) {
      if (unused_snvs.size() > 0) {

        return CN_LABEL_WEIGHT * get_sample_cn_label_log_kernel() + std::log(1.0 - std::exp(CN_LABEL_WEIGHT)) * get_sample_snv_label_log_kernel();
      }
      return get_sample_cn_label_log_kernel();
    }
    return get_sample_snv_label_log_kernel();
  }

  TreeLabel sample_label(Random<Real_t> &random) {
    if (random.uniform() <= std::exp(CN_LABEL_WEIGHT) && has_free_cn_labels()) {
      return sample_cn_label(random);
    }
    return sample_snv_label(random);
  }

  bool has_free_labels() { return has_free_cn_labels() || has_free_snv_labels(); }
};

#endif // !VERTEX_SAMPLER_H
