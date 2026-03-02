/*
 @file
 @brief - GraphDef event writer
 @note - see schema.h for details
 */
#pragma once

#include "writer.h"

namespace tensorboard {

// ── GraphDef ───────────────────────────────────────────────────────────
struct Node : public proto::Encoder {
    explicit Node(const STR& name, const STR& op) {
        str(1, name);
        str(2, op);
    }
    explicit Node(const STR& name, const STR& op, const STR& input) {
        str(1, name);
        str(2, op);
        if (input.size() > 0) str(3, input);
    }
    void add_input(const STR& input) {
        if (input.size() > 0) str(3, input);
    }
};

class GraphWriter : public EventWriter {
    std::vector<Node> _net;
        
public:
    explicit GraphWriter(const STR& path) : EventWriter(path) {}

    void add_node(const Node& node) {
        _net.push_back(node);
    }

    void write(S64 step=0) {
        proto::Encoder graph;
        for (auto n : _net) {
            graph.raw(1, n.buf());
        }
        
        proto::Encoder event;
        event.s64(2, step);
        event.raw(4, graph.buf());
        
        _write(event.buf());
    }
};

} // class Node
