#ifndef TREE_FORMATTER_H
#define TREE_FORMATTER_H

#include <map>
#include <sstream>
#include <string>
#include <string_view>

#include "event_tree.h"

class TreeFormatter {
  using NodeHandle = EventTree::NodeHandle;

  static std::string get_root_string_rep() { return "(0,0)"; }

  static std::string get_node_label(EventTree &tree, NodeHandle node) {
    return node == tree.get_root() ? get_root_string_rep()
                                   : label_to_str(node->label);
  }

  static void to_string(EventTree &tree, NodeHandle node,
                        std::stringstream &ss) {
    for (auto &child : node->children) {
      ss << get_node_label(tree, node) << "-" << get_node_label(tree, child)
         << "\n";
      to_string(tree, child, ss);
    }
  }

public:
  static std::string to_string_representation(EventTree &tree) {
    std::stringstream ss;
    to_string(tree, tree.get_root(), ss);
    return ss.str();
  }
};
#endif