#ifndef ATTACHMENT_ENRICHER_H
#define ATTACHMENT_ENRICHER_H
#include <algorithm>
#include <tuple>
#include <vector>

#include "conet_result.h"
#include "tree/attachment.h"
#include "tree/tree_counts_scoring.h"
#include "tree/tree_formatter.h"
#include "utils/logger/logger.h"
#include "utils/utils.h"






template <class Real_t> class AttachmentEnricher {
public:
  using NodeHandle = EventTree::NodeHandle;
  using MoveData = typename MHStepsExecutor<Real_t>::MoveData;
  EventTree &tree;
  CountsDispersionPenalty<Real_t> dispersion_penalty_calculator;
  Random<Real_t> random;
  Attachment attachment;
  Real_t current_penalty = 0.0;
  std::map<TreeLabel, NodeHandle> label_to_node;
  size_t accepted = 0;

public:
  AttachmentEnricher(EventTree &tree,
                        Attachment attachment,
                         unsigned int seed,
                         CONETInputData<Real_t> &cells)
      : tree{tree},
        dispersion_penalty_calculator{cells},
        random{seed}, attachment{attachment}{
            current_penalty = dispersion_penalty_calculator.calculate_log_score(tree, attachment);
            for (auto node: tree.get_descendants(tree.get_root())) {
                label_to_node[tree.get_node_label(node)] = node;
            }
        }

   void iter() {
    auto cell = random.next_int(attachment.cell_to_tree_label.size());
    auto attachment_node = label_to_node[attachment.cell_to_tree_label[cell]];
    std::vector<NodeHandle> nodes;
    for (auto child: tree.get_children(attachment_node)) {
        nodes.push_back(child);
    }
    auto parent = tree.get_parent(attachment_node);
    if (parent != nullptr) {
        nodes.push_back(parent);
    }
    auto new_node_attachment = nodes[random.next_int(nodes.size())];
    attachment.cell_to_tree_label[cell] = tree.get_node_label(new_node_attachment);
    auto new_counts_penalty = dispersion_penalty_calculator.calculate_log_score(tree, attachment);
    if (new_counts_penalty < current_penalty) {
        attachment.cell_to_tree_label[cell] = tree.get_node_label(attachment_node);
    } else {
        current_penalty = new_counts_penalty;
        accepted += 1;
    }
   }
};

#endif