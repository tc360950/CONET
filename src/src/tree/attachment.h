#ifndef ATTACHMENT_H
#define ATTACHMENT_H
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

#include "../types.h"

/**
 * @brief Attachment of cells to Event Tree nodes
 *
 */
class Attachment {
public:
  std::vector<TreeLabel> cell_to_tree_label;

  Attachment(TreeLabel default_label, size_t cells) {
    for (size_t i = 0; i < cells; i++) {
      cell_to_tree_label.push_back(default_label);
    }
  }

  bool has_attached_cells(TreeLabel label) const {
    return std::find(cell_to_tree_label.begin(), cell_to_tree_label.end(),
                     label) != cell_to_tree_label.end();
  }

  void set_attachment(size_t cell, TreeLabel node) {
    cell_to_tree_label[cell] = node;
  }

  std::map<TreeLabel, std::set<size_t>> get_node_label_to_cells_map() const {
    std::map<TreeLabel, std::set<size_t>> node_to_cells;
    for (size_t cell = 0; cell < cell_to_tree_label.size(); cell++) {
      if (node_to_cells.find(cell_to_tree_label[cell]) == node_to_cells.end()) {
        node_to_cells[cell_to_tree_label[cell]] = std::set<size_t>();
      }
      node_to_cells[cell_to_tree_label[cell]].insert(cell);
    }
    return node_to_cells;
  }

  friend std::ostream &operator<<(std::ostream &stream,
                                  const Attachment &attachment) {
    for (size_t j = 0; j < attachment.cell_to_tree_label.size(); j++) {
      if (is_cn_event(attachment.cell_to_tree_label[j])) {
          auto label_ = get_event_from_label(attachment.cell_to_tree_label[j]);
          stream << j << ";" << label_.first << ";" << label_.second << "\n";
      } else {
        auto label_ = std::get<1>(attachment.cell_to_tree_label[j]);
        stream << j << ";SNV;" << label_.snv << "\n";
      }
    }
    return stream;
  }

  friend void swap(Attachment &a, Attachment &b) {
    std::swap(a.cell_to_tree_label, b.cell_to_tree_label);
  }
};
#endif