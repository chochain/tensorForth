/** -*- c++ -*-
 * @file
 * @brief AIO class - async IO module implementation
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <cstdio>        // printf
#include <iostream>      // cin, cout
#include <iomanip>       // setbase, setprecision
#include "aio.h"

#if (T4_ENABLE_OBJ && T4_ENABLE_NN)
///
/// NN Model IO private methods
///
__HOST__ void
AIO::_print_model(Model &m) {
    auto tinfo = [this, &fout](Tensor &t, int i, int fn) { ///> layer info
        fout << "[" << std::setw(3) << i << "] "
             << Model::nname(fn) << ":";
        to_s(fout, t, false);
        int sz = 0;
        for (int n = 0; n < 4; n++) {
            sz += t.grad[n] ? t.grad[n]->numel : 0;
        }
        fout << ", #param=" << sz;
    };
    auto finfo = [this, &fout](Tensor **g) {
        for (int i=0; g[i] && i < 2; i++) {
            fout << " "; to_s(fout, *g[i], false);
        }
    };
    if (!m.is_model()) return;
    int n = m.numel;
    
    fout << "NN model[" << n-1 << "/" << m.slots() << "]" << endl;
    for (int i = 1; i < n; i++) {              /// skip root[0]
        Tensor &in = m[i], &out = m[i+1];
        tinfo(in, i, in.grad_fn);
        finfo(in.grad);
        _print_model_parm(fout, in, out);
        fout << endl;
    }
}
///
/// print model layer parameters
///
__HOST__ void
AIO::_print_model_parm(Tensor &in, Tensor &out) {
    t4_layer fn = in.grad_fn;             ///< layer function
    DU       p  = 0.001 * in.parm;        ///< layer parameter
    switch(fn) {
    case L_NONE:    /* do nothing  */                  break;
    case L_CONV:   fout << " bias=" << p << ", C="
                        << out.C() << " ";             break;
    case L_LINEAR: fout << " bias=" << p << ", H="
                        << in.grad[0]->H() << " ";     break;
    case L_FLATTEN:
    case L_RELU:
    case L_TANH:
    case L_SIGMOID: /* do nothing */                   break;
    case L_SELU:
    case L_LEAKYRL:
    case L_ELU:     fout << " bias=" << p << " ";      break;
    case L_DROPOUT: fout << " rate=" << p*100.0 << "% "; break;
    case L_SOFTMAX:
    case L_LOGSMAX: /* do nothing */                   break;
    case L_AVGPOOL:
    case L_MAXPOOL:
    case L_MINPOOL: fout << " n=" << in.parm << " ";   break;
    case L_BATCHNM: fout << " mtum=" << p << " ";      break;
    case L_USAMPLE: {
        const char *nm[] = { "nearest", "linear", "bilinear", "cubic" };
        int n = in.parm & 0xff;
        fout << " " << n << "x" << n << " " << nm[in.parm>>8];
    } break;
    default: fout << "unknown layer=" << fn << " ";    break;
    }
}
///
/// initial dataset setup
/// init flow:
///    netvm#dataset
///    -> aio::process_node
///    -> mmu::dataset          - set N=batch_sz, batch_id = -1
///
/// fetch flow:
///    netvm#fetch
///    -> aio::process_node
///    -> aio::fetch
///      -> corpus::fetch       - fetch host label/image blocks from files
///      -> dataset::reshape    - set dimensions for the first batch (no alloc yet) 
///      -> dataset::load_batch - transfer host blocks to device
///         -> dataset::alloc   - alloc device memory blocks if needed
/// Note:
///   ds_name: dataset name (match in loader.cu), for initial dataset setup
///   ds_name: NULL, following batch
///
__HOST__ int
AIO::_dsfetch(Dataset &ds, U16 mode, char *ds_name) {
//    Dataset &ds = (Dataset&)_mmu->du2obj(id);     ///< dataset ref
//    U32     dsx = DU2X(id) & ~T4_TYPE_MSK;        ///< dataset mnemonic
//    if (!ds.is_dataset()) {                       /// * indeed a dataset?
//        ERROR("mmu#load id=%x is not a dataset\n", dsx);
//        return -1;
//    }
    bool rewind = (mode & FAM_REW) != 0;
    ///
    /// search cache for top <=> dataset pair
    ///
    IO_TRACE("\nAIO::%s dataset (id=%x) =>",
           ds_name ? ds_name : (rewind ? "rewind" : "fetch"), dsx);
    Corpus *cp = Loader::get(dsx, ds_name);      ///< Corpus/Dataset provider
    if (!cp) {
        ERROR(" dataset not found\n"); return -1;
    }
    int batch_sz = ds.N();                       ///< mini batch size
    if (ds_name) {                               /// * init load
        if (cp->init()==NULL) {
            ERROR(" dataset setup failed!\n"); return -2;
        }
        ds.reshape(batch_sz, cp->H, cp->W, cp->C);/// * reshape ds to match Corpus
    }
    if (rewind) {
        cp->rewind();
        ds.batch_id = ds.done = 0;
    }
    else if ((ds.done=cp->eof)) {                /// * dataset exhausted?
        IO_TRACE(" completed, no more data.\n"); return 0;
    }
    ///
    /// load a mini-batch of data points
    ///
    if (!cp->fetch(ds.batch_id, batch_sz)) {     /// * fetch a batch from Corpus
        ERROR("fetch failed\n");  return -3;
    }
    ///
    /// transfer host into device memory
    /// if needed, allocate Dataset device (managed) memory blocks
    ///
    ds.load_batch(cp->data, cp->label);
    IO_TRACE("batch[%d] %d record(s) loaded\n", ds.batch_id, batch_sz);

    ds.batch_id++;
    ds.done = cp->eof;
    
    return 0;
}
///
/// NN model persistence (i.e. serialization) methods
///
#include <fstream>
__HOST__ int
AIO::_nsave(Model &m, U16 mode, char* fname) {
    printf("\nAIO::save model to '%s' =>", fname);
//    Model &m = (Model&)_mmu->du2obj(top);
    ofstream fout(fname, ios_base::binary);     ///< open an output file
    if (!fout.is_open()) {
        ERROR(" failed to open for output\n");
        return 1;
    }
    fout << "\\ " << T4_APP_NAME << " model\n\\ version v" << T4_MAJOR_VER << "." << T4_MINOR_VER << "\n";
    if (mode & FAM_RAW) {
        // TODO: raw format (.npy, .petastorm, hdf5)
    }
    else {
        _nsave_model(fout, m);                  /// * blank line as section break
        _nsave_param(fout, m);
    }
    fout << "\n---" << endl;
    fout.close();
    printf(" completed\n");
    return 0;
}

__HOST__ int
AIO::_nload(Model &m, U16 mode, char* fname) {
    printf("\nAIO::load '%s' ", fname);
//    Model &m = (Model&)_mmu->du2obj(top);
    ifstream fin(fname, ios_base::binary);           ///< open an input file
    if (!fin.is_open()) {
        ERROR("=> failed to open for input\n");
        return 1;
    }
    /// TODO: handle raw data format
    int err = 0;
    if (m.numel <= 2) {
        printf("NN model");
        err = _nload_model(fin, m, fname);           /// * load model layers
    }
    else {
        std::string tmp;
        while (getline(fin, tmp) && tmp.length());   /// * skip model section
        printf("parameter tensors (i.e. state_dict)");
        err = _nload_param(fin, m);           /// * load model layer tensors
    }
    fin.close();
    printf(" => %s\n", err ? "error" : "completed");
    return err;
}

__HOST__ int
AIO::_nsave_model(Model &m) {
    for (U16 i = 1; i < m.numel - 1; i++) {
        Tensor &in = m[i], &out = m[i+1];
        _print_model_parm(fout, in, out);
        
        const char *nm = Model::nname(in.grad_fn);
        fout << nm << endl;                   /// * one blank line serves
                                              /// * as the sectional break
        printf("\n%2d> %s [%d,%d,%d,%d]\tp=%-2d => out[%d,%d,%d,%d]",
            i, nm, in.N(), in.H(), in.W(), in.C(), in.parm,
            out.N(), out.H(), out.W(), out.C());
    }
    return 0;
}

__HOST__ int
AIO::_nsave_param(Model &m) {
    auto _dump = [&fout](const char pn, const char *nm, Tensor &t) {
        fout << "\n--- " << pn << "." << nm << endl;     /// * section marker
        fout.write((char*)t.data, t.numel * sizeof(DU));
    };
    for (U16 i = 1; i < m.numel - 1; i++) {
        Tensor   &in = m[i];                             ///< nth model layer
        t4_layer fn  = in.grad_fn;                       ///< layer function
        const char *nm = Model::nname(fn);               ///< layer name
        switch(fn) {
        case L_CONV:
        case L_LINEAR:
            _dump('w', nm, *in.grad[0]);
            _dump('b', nm, *in.grad[1]);  break;
        case L_BATCHNM:
            _dump('w', nm, *in.grad[0]);  break;
        default: break;
        }
    }
    return 0;
}

__HOST__ int
AIO::_nload_model(Model &m, char *fname) {
    std::string line;
    while (getline(fin, line) && line[0] == '\\') {    /// * TODO: check version
        cout << endl << line;
    }
    if (m.numel > 2) return 0;                         /// * model already loaded

    std::string cmd = line;                            ///< input command
    while (getline(fin, line) && line.length()) {      /// * append layer-by-layer
        cout << endl << line;                          /// * til blank line as break
        cmd.append(" " + line);
    }
    cmd.append(" nn.load ").append(fname);             /// * add parameter reload command
    
    _istr->clear();                                    /// * setup input command buffer
    char *tib = _istr->rdbuf();
    if (cmd.length() >= T4_IBUF_SZ) {                  /// * check buffer size
        *tib = '\0';
        ERROR(" input buffer (T4_IBUF_SZ) overflow!\n");
        return 1;
    }
    strcpy(tib, cmd.c_str());                          /// * fill buffer from model file
    return 0;
}

__HOST__ int
AIO::_nload_param(Model &m) {
    auto _read = [&fin](const char *pn, const char *nm, Tensor &t) {
        std::string line;                              ///< input string
        printf("\n%s %s[%d,%d,%d,%d] ", nm, pn, t.N(), t.H(), t.W(), t.C());
        while (getline(fin, line) && !line.length());  /// * skip blank lines
        if (line[0]!='-' || line[1]!='-' || line[2]!='-') {
            printf(" model format error");
            return 1;
        }
        fin.read((char*)t.data, t.numel * sizeof(DU)); /// * load parameters
        printf("= %ld bytes", fin.gcount());
        return 0;
    };
    for (int i = 1; i < m.numel - 1; i++) {
        Tensor  &in = m[i];                            ///< layer tensor
        t4_layer fn = in.grad_fn;                      ///< layer function
        const char *nm = Model::nname(fn);
        switch(fn) {
        case L_CONV:
        case L_LINEAR:
            _read("W", nm, *in.grad[0]);
            _read("B", nm, *in.grad[1]); break;
        case L_BATCHNM:
            _read("W", nm, *in.grad[0]); break;
        default: break;
        }
    }
    return 0;
}

#endif // (T4_ENABLE_OBJ && T4_ENABLE_NN)
