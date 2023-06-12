#ifndef COLORING_H
#define COLORING_H

#include "tree/attachment.h"
#include "tree/event_tree.h"

#include <utility>
#include <list>

template <class Real_t> class MaxTreeColoring {
public:
    using NodeHandle = EventTree::NodeHandle;
    /**
        Each node gets an arbitrary value (may be negative).
        Find maximal tree coloring such that every leaf-root path has
        at most one colored node.

        Score of a coloring is equal to sum of colored nodes' values.
        Score of tree without colored nodes is equal to 0.
    **/


    std::pair<Real_t, std::list<NodeHandle>> get_max_colored_score(NodeHandle node, EventTree& tree,std::map<NodeHandle, Real_t> &node_values) {
    // Get max score obtained from coloring of subtree rooted at @node and list of colored nodes
        Real_t children_score = 0.0;

        std::list<NodeHandle> children_colored;
        for (auto child: tree.get_children(node)) {
            auto score_colored = get_max_colored_score(child, tree, node_values);
            children_colored.merge(std::get<1>(score_colored));
            children_score += std::get<0>(score_colored);
        }

        if (node == tree.get_root()) {
            return std::make_pair(children_score, children_colored);
        }

        if (node_values[node] <= 0.0 && children_score <= 0.0) {
                return std::make_pair(0.0, std::list<NodeHandle>());
        }

        if (children_score < node_values[node]) {
            return std::make_pair(node_values[node], std::list{node});
        }
        return std::make_pair(children_score, children_colored);
    }

};
#endif