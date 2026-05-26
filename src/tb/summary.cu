/** -*- c++ -*-
 * @file
 * @brief Tensorboard class - async IO module implementation
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "mu/tensor.h"
#include "nn/model.h"
#include "summary.h"

namespace t4::tb {

#if T4_DO_TB

__HOST__ void
Summary::init(const char *run_id) {
    if (run_id && strcmp(run_id, _run_id) != 0) {
        teardown();
        _run_id = run_id;
    }
    std::string rundir  = std::string(_root) + "/" + _run_id;
    std::string logname = _logname(rundir);
    mkdir(rundir.c_str(), 0755);                  /// * create Event/Run subdir
    EventWriter::setup(logname.c_str());
}

__HOST__ void
Summary::image(const char *tag, T4Base &b) {
    if (!(b.is_tensor() || b.is_dataset())) {
        ERROR("summary#image requires tensor or dataset (b.ttype=%d)\n", b.ttype);
        return;
    }
    Tensor &t = (Tensor&)b;
    const U32 W = t.W(), H = t.H(), C = t.C();
    const DU  mean = t.avg(), scale = (t.std() - 0.5f) * 128.0f;  /// 95%
    U8V px(W * H * 3);
    DU  hx[W * H * C];
    for (int n = 0; n < t.N(); n++) {
        DU *d = t.slice(n), *h = hx;
        D2H(h, d, sizeof(DU) * W * H * C);
        for (int y = 0; y < H; y++) {
            U8 *p = &px[(y * W) * 3];
            for (int x = 0; x < W; x++, h++) {
                DU vx = (*h - mean) * scale + 128.5f;
                U8 v  = (U8)MIN(255, MAX(vx, 0));
                *p++ = v;
                *p++ = v;
                *p++ = v;
            }
        }
        add_image(tag, W, H, px, _step);
    }
}

__HOST__ void
Summary::tile(const char *tag, T4Base &b, int n_per_row) {
    if (!(b.is_tensor() || b.is_dataset())) {
        ERROR("summary#tile requires tensor or dataset (b.ttype=%d)\n", b.ttype);
        return;
    }
    Tensor &t = (Tensor&)b;
    const U32  N     = t.N(), H  = t.H(), W = t.W(), C = t.C();
    const int  WT    = n_per_row * W;
    const int  HT    = (N + n_per_row - 1) / n_per_row;
    const DU   mean  = t.avg();
    const DU   scale = (64.0f / t.std());        /// 2 std = 95%

    auto tile = [&](U8V &px, DU *v, int idx) {
        int ht = idx / n_per_row, wt = idx % n_per_row;
        U8 *p = &px[(ht * H * WT + wt * W) * 3];
        for (int y = 0; y < H; y++) {
            for (int x = 0, c = 0; x < W; x++, c=0) {
                while (c < 3) {                  /// RGB
                    DU vx = (*v - mean) * scale + 128.0f;
                    *p++ = (x==0 && y==0)
                        ? 128 : static_cast<U8>(MIN(255, MAX(vx, 0)));
                    if (c++ < C) v++;            /// advance if more than 1 channel
                }
            }
            p += (WT - W) * 3;                   /// skip to next row in tile
        }
    };

    U8V px((HT * H) * WT * 3);                  ///< zero-init, so unfilled are black
    DU  h[t.numel];                             ///< host block (watch out, heap space)
    D2H(h, t.data, sizeof(DU) * t.numel);       /// * copy to host
    for (int n = 0; n < N; n++) {
        DU *hx = &h[n * H * W * C];
        tile(px, hx, n);                         /// * fill tile by tile
    }
    add_image(tag, WT, H * HT, px, _step);
}

__HOST__ void
Summary::histo(const char *tag, T4Base &b, int n_bucket) {
    if (!b.is_tensor()) {
        ERROR("summary#histo requires tensor (b.ttype=%d)\n", b.ttype);
        return;
    }
    Tensor &t = (Tensor&)b;
    DU tx[t.numel];
    D2H(tx, t.data, sizeof(DU) * t.numel);
    add_histo(tag, tx, t.numel, _step, n_bucket);
}
///
/// print model layer parameters
///    
/// dataflow op: Placeholder, Const, Variable, Variable2, VarHandleOp
/// structure op: NoOp, Assign, AssignAdd, AssignSub
/// math op: Add, BiasAdd, Sub, Mul, Div, RealDiv
/// nn   op: Conv2D, Maxpool, AvgPool, Softmax
/// act  op: Relu, Relu6, Sigmoid, Tanh
/// Node(name, op, input)
///    
__HOST__ void
Summary::graph(T4Base &b) {
    if (!b.is_model()) {
        ERROR("summary#graph requires model (b.ttype=%d)\n", b.ttype);
        return;
    }
    const char *op[] = {
        "Output", "Conv2D", "MatMul", "Reshape",
        "Relu", "Tanh", "Sigmoid", "Selu", "LeakyRelu", "Elu",
        "Dropout", "Softmax", "LogSoftmax",
        "AvgPool", "MaxPool", "MinPool", "BatchNorm", "UpSample"
    };
    auto _tname = [op](Tensor &t, int i) {
        std::ostringstream ss;
        ss << op[t.grad_fn] << "_" << i << "/" << Model::nname(t.grad_fn);
        return ss.str();
    };
    auto _node_attr = [](graph::Node &n, Tensor &t) {
        U32V s = { t.N(), t.H(), t.W(), t.C() };
        n.add_type("dtype", 1);
        n.add_shape(s);
    };
    
    Model &m = (Model&)b;
    init_graph();
    
    graph::Node nn[m.numel];                    ///< allocate nodes
    graph::Node n0("input", "Placeholder", ""); ///< input node
    _node_attr(n0, m[0]);
    add_node(n0);

    for (int i = 0; i < m.numel; i++) {
        graph::Node &n = nn[i];
        Tensor &in = m[i];
        std::string nm  = _tname(in, i);
        std::string nm0 = i==0 ? std::string("input") : _tname(m[i-1], i-1);
        
        INFO("%s <= %s fn=%d\n", nm0.c_str(), nm.c_str(), in.grad_fn);
        
        n.init(nm.c_str(), op[in.grad_fn], nm0.c_str());
        _node_attr(n, in);
        
        add_node(n);
    }
    add_graph();
}

typedef std::string STR;

__HOST__ void
Summary::embed(const char* tag, T4Base &b) {
    if (!b.is_tensor()) {
        ERROR("summary#embed requires tensor (b.ttype=%d)\n", b.ttype);
        return;
    }
    Tensor &t = (Tensor&)b;
    F32    v[t.numel];
    D2H(v, t.data, sizeof(DU) * t.numel);         /// * move from device to host
    
    std::string rundir = std::string(_root) + "/" + _run_id;
    const char *path   = rundir.c_str();
    
    _proj.add_embedding(path, tag, v, t.N(), t.HWC());
    _proj.flush_config(path);                     /// * safe to call repeatedly — overwrites
}
#endif // T4_DO_TB

} // namespace t4::tb
