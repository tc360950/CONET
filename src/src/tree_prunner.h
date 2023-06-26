#ifndef TREE_PRUNNER_H
#define TREE_PRUNNER_H
#include <algorithm>
#include <tuple>
#include <vector>

#include "conet_result.h"
#include "tree/attachment.h"
#include "tree/tree_counts_scoring.h"
#include "tree/tree_formatter.h"
#include "utils/logger/logger.h"
#include "utils/utils.h"





template <class Real_t> class TreePrunner {
public:
  using NodeHandle = EventTree::NodeHandle;
  EventTree &tree;
  CONETInputData<Real_t> &cells;
  Attachment attachment;
  Real_t min_coverage = 0.0;
  std::map<TreeLabel, NodeHandle> label_to_node;

public:
  TreePrunner(EventTree &tree,
                        Attachment attachment,
                         CONETInputData<Real_t> &cells, Real_t min_coverage)
      : tree{tree},
        cells{cells},
        attachment{attachment}, min_coverage{min_coverage}{
            for (auto node: tree.get_descendants(tree.get_root())) {
                label_to_node[tree.get_node_label(node)] = node;
            }
        }

   std::vector<NodeHandle> get_leaves() {
    auto nodes = tree.get_descendants(tree.get_root());
    std::vector<NodeHandle> leaves;
    for (auto n : nodes) {
        if (tree.is_leaf(n) && n != tree.get_root()) {
            leaves.push_back(n);
        }
    }
    return leaves;
   }

   Real_t get_node_coverage(NodeHandle node) {
        auto label = tree.get_node_label(node);
        Real_t result = 0.0;
        for (size_t cell = 0; cell < attachment.cell_to_tree_label.size(); cell++) {
            if (attachment.cell_to_tree_label[cell] == label) {
                Real_t sum = std::accumulate(cells.D[cell].begin(), cells.D[cell].end(), 0.0);
                sum = sum / cells.D[cell].size();
                result += sum;
            }
        }
        return result;
   }

   void prune_node(NodeHandle node) {
    auto parent_label = tree.get_node_label(tree.get_parent(node));
    auto child_label = tree.get_node_label(node);
    for (size_t cell = 0; cell < attachment.cell_to_tree_label.size(); cell++) {
            if (attachment.cell_to_tree_label[cell] == child_label) {
                attachment.set_attachment(cell, parent_label);
            }
    }
    tree.delete_leaf(node);
   }

   bool iter() {
    auto leaves = get_leaves();
    std::map<NodeHandle, Real_t> leaf_to_coverage;
    Real_t l_min_coverage = -1;
    NodeHandle min_leaf = nullptr;
    for (auto l : leaves) {
        leaf_to_coverage[l] = get_node_coverage(l);
        if (min_leaf == nullptr || leaf_to_coverage[l] < l_min_coverage) {
            l_min_coverage = leaf_to_coverage[l];
            min_leaf = l;
        }
    }
    if (l_min_coverage >= min_coverage || min_leaf == nullptr) {
        return false;
    }
    log("Min coverage in leaf clusters: ", l_min_coverage, " ;will prune this leaf. Tree size: ", tree.get_size());
    prune_node(min_leaf);
    return true;
   }
};

#endif