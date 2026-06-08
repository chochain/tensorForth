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
    STR rundir  = STR(_root) + "/" + esc(_run_id);
    STR logname = _logname(rundir);

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
    const int N = t.N(), W = t.W(), H = t.H(), C = t.C();
    const DU mean = t.avg(), scale = (t.std() - 0.5f) * 128.0f;  /// 95%
    
    U8V  px(H * W * 3);
    F32V hx(H * W * C);
    for (int n = 0; n < N; n++) {
        DU *d = t.slice(n), *h = hx.data();
        D2H(h, d, sizeof(DU) * W * H * C);
        for (int y = 0; y < H; y++) {
            U8 *p = &px[(y * W) * 3];
            for (int x = 0; x < W; x++, h++) {
                for (int c = 0; c < MAX(C, 3); c++) {
                    DU vx = (*h + mean) * scale;
                    U8 v  = (U8)MIN(255.0f, MAX(vx, 0));
                    *p++ = v;
                    if (c < C) h++;
                }
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
    const int  N    = t.N(), H = t.H(), W = t.W(), C = t.C();
    const int  WT   = W * n_per_row;
    const int  HT   = H * ((N + n_per_row - 1) / n_per_row);
    const F32  mean = 0.0f, scale = 256.0f;
//    const F32  mean  = t.avg(), scale = 64.0f / t.std();        /// 2 std = 95%

    U8V  px(HT * WT * 3);                         ///< zero-init, so unfilled are black
    F32V hx(t.numel);                             ///< host block on heap (CC: watch)
    D2H(hx.data(), t.data, t.numel * sizeof(DU)); /// * copy device to host

    for (int n = 0; n < N; n++) {                 ///< CC TODO: in worker thread
        F32 *v = &hx[n * H * W * C];              ///< or pre-build px in kernel
        int ty = n / n_per_row, tx = n % n_per_row;
        for (int y = 0; y < H; y++) {
            U8 *p = px.data() + ((ty * H + y) * WT + tx * W) * 3;
            for (int x = 0; x < W; x++) {
                if (x==0 || x==(W-1) || y==0 || y==(H-1)) {       /// * boarder
                    p[0] = 196; p[1] = 160; p[2] = 160;
                    p += 3;
                }
                else {
                    for (int c = 0; c < 3; c++) {
                        DU vx = (*(v + (c < C ? c : C-1)) + mean) * scale;
                        *p++ = static_cast<U8>(MIN(255.0f, MAX(vx, 0.0f)));
                    }
                }
                v += C;
            }
        }
    }
    add_image(tag, WT, HT, px, _step);
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
        ss <<  op[t.grad_fn] << "_" << i << "/" << Model::nname(t.grad_fn);
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

__HOST__ void
Summary::embed(const char* tag, T4Base &b) {
    if (!b.is_tensor()) {
        ERROR("summary#embed requires tensor (b.ttype=%d)\n", b.ttype);
        return;
    }
    Tensor &t = (Tensor&)b;
    F32    v[t.numel];
    D2H(v, t.data, sizeof(DU) * t.numel);         /// * move from device to host
    
    STR rundir       = STR(_root) + "/" + esc(_run_id);
    const char *path = rundir.c_str();
    
    _proj.add_embedding(path, esc(tag).c_str(), v, t.N(), t.HWC());
    _proj.flush_config(path);                     /// * safe to call repeatedly — overwrites
}

#endif // T4_DO_TB

} // namespace t4::tb
