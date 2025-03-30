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

#if (T4_DO_OBJ && T4_DO_NN)
#include <fstream>
#include "nn/dataset.h"
#include "nn/model.h"
#include "ldr/loader.h"  // includes Corpus

using namespace std;
///
/// initial dataset setup
/// init flow:
///    netvm::dataset
///    -> sys::process_event
///    -> nn::dataset           - set N=batch_sz, batch_id = -1
///
/// fetch flow:
///    netvm::fetch
///    -> sys::process_event
///    -> aio::dsfetch
///      -> corpus::fetch       - fetch host label/image blocks from files
///      -> dataset::reshape    - set dimensions for the first batch (no alloc yet) 
///      -> dataset::load_batch - transfer host blocks to device
///         -> dataset::alloc   - alloc device memory blocks if needed
/// Note:
///   ds_name: dataset name (match in loader.cu), for initial dataset setup
///   ds_name: NULL, following batch
///
__HOST__ int
AIO::dsfetch(Dataset &ds, char *ds_name, bool rewind) {
    static const char *fn = "aio#dsfetch";
    ///
    /// search cache for top <=> dataset pair
    ///
    IO_DB("\n  %s(%s) dataset (batch_id=%d) {\n",
          fn, ds_name ? ds_name : (rewind ? "rewind" : "fetch"), ds.batch_id);
    Corpus *cp = Loader::get(ds, ds_name);       ///< Corpus/Dataset provider
    if (!cp) {
        ERROR("  } %s => dataset not found\n", fn); return -1;
    }
    int batch_sz = ds.N();                       ///< mini batch size
    if (ds_name) {                               /// * init load
        if (cp->init(trace)==NULL) {
            ERROR("  } %s => dataset setup failed!\n", fn); return -2;
        }
        ds.reshape(batch_sz, cp->H, cp->W, cp->C);/// * reshape ds to match Corpus
    }
    if (rewind) {
        cp->rewind();
        ds.batch_id = ds.done = 0;
    }
    else if ((ds.done=cp->eof)) {                /// * dataset exhausted?
        IO_DB("  } %s => completed, no more data.\n", fn); return 0;
    }
    ///
    /// load a mini-batch of data points
    ///
    if (!cp->fetch(ds.batch_id, batch_sz, trace)) {     /// * fetch a batch from Corpus
        ERROR("  } %s => fetch failed\n", fn);  return -3;
    }
    ///
    /// transfer host into device memory
    /// if needed, allocate Dataset device (managed) memory blocks
    ///
    ds.load_batch(cp->data, cp->label);
    IO_DB("  } %s => batch[%d] %d record(s) loaded, done=%d\n",
          fn, ds.batch_id, batch_sz, cp->eof);

    ds.batch_id++;
    ds.done = cp->eof;
    
    return 0;
}
///
/// NN model persistence (i.e. serialization) methods
///
__HOST__ int
AIO::nsave(Model &m, char* fname, U8 mode) {
    IO_DB("\nAIO::save model to '%s' {\n", fname);
    ofstream fs(fname, ios_base::binary);     ///< open an output file
    if (!fs.is_open()) {
        ERROR("} => failed to open for output\n");
        return 1;
    }
    fs << "\\ " << T4_APP_NAME << " model\n";
    if (mode & FAM_RAW) {
        // TODO: raw format (.npy, .petastorm, hdf5)
    }
    else {
        _nsave_model(fs, m);                  /// * blank line as section break
        _nsave_param(fs, m);
    }
    fs << "\n---" << std::endl;
    fs.close();
    IO_DB("} => completed\n");
    return 0;
}

__HOST__ int
AIO::nload(Model &m, char* fname, U8 mode, char *tib) {
    IO_DB("\nAIO::load '%s' {\n", fname);
    ifstream fs(fname, ios_base::binary);            ///< open an input file
    if (!fs.is_open()) {
        ERROR("} => failed to open for input\n");
        return 1;
    }
    /// TODO: handle raw data format
    int err = 0;
    if (m.numel <= 2) {
        IO_DB("NN model");
        err = _nload_model(fs, m, fname, tib);      /// * load model layers
    }
    else {
        std::string tmp;
        while (getline(fs, tmp) && tmp.length());   /// * skip model section
        IO_DB("  parameter tensors (i.e. state_dict)");
        err = _nload_param(fs, m);                  /// * load model layer tensors
    }
    fs.close();
    IO_DB("} => %s\n", err ? "error" : "completed");
    return err;
}
///
/// NN Model IO private methods
///
__HOST__ void
AIO::_print_model(h_ostr &fs, Model &m) {
    auto tinfo = [this,&fs](Tensor &t, int i, int fn) { ///> layer info
        fs << "[" << std::setw(3) << i << "] "
           << Model::nname(fn) << ":";
        to_s(fs, t, false);
        int sz = 0;
        for (int n = 0; n < 4; n++) {
            sz += t.grad[n] ? t.grad[n]->numel : 0;
        }
        fs << " #p=" << sz << ' ';
    };
    auto finfo = [this,&fs](Tensor **g) {
        for (int i=0; g[i] && i < 2; i++) {
            to_s(fs, *g[i], false); fs << ' ';
        }
    };
    if (!m.is_model()) return;
    U64 n = m.numel;
    
    fs << "NN model[" << n-1 << "/" << m.slots() << "]"
       << std::endl;
    for (U64 i = 1; i < n; i++) {         /// skip root[0]
        Tensor &in = m[i], &out = m[i+1];
        tinfo(in, (int)i, in.grad_fn);
        finfo(in.grad);
        _print_model_parm(fs, in, out);
        fs << std::endl;
    }
}
///
/// print model layer parameters
///
__HOST__ void
AIO::_print_model_parm(h_ostr &fs, Tensor &in, Tensor &out) {
    t4_layer fn = in.grad_fn;             ///< layer function
    DU       p  = in.xparm;               ///< layer parameter
    switch(fn) {
    case L_NONE:    /* do nothing  */                  break;
    case L_CONV:   fs << "bias=" << p << ", C="
                      << out.C();                      break;
    case L_LINEAR: fs << "bias=" << p << ", H="
                      << in.grad[0]->H();              break;
    case L_FLATTEN:
    case L_RELU:
    case L_TANH:
    case L_SIGMOID: /* do nothing */                   break;
    case L_SELU:
    case L_LEAKYRL:
    case L_ELU:     fs << "bias=" << p;                break;
    case L_DROPOUT: fs << "rate=" << p*100.0 << '%';   break;
    case L_SOFTMAX:
    case L_LOGSMAX: /* do nothing */                   break;
    case L_AVGPOOL:
    case L_MAXPOOL:
    case L_MINPOOL: fs << "n=" << in.iparm;            break;
    case L_BATCHNM: fs << "mtum=" << p;                break;
    case L_USAMPLE: {
        const char *nm[] = { "nearest", "linear", "bilinear", "cubic" };
        int n = in.iparm & 0xff;
        fs << n << "x" << n << " " << nm[in.iparm>>8];
    } break;
    default: fs << "unknown layer=" << fn;      break;
    }
}
__HOST__ int
AIO::_nsave_model(h_ostr &fs, Model &m) {
    for (U16 i = 1; i < m.numel - 1; i++) {
        Tensor &in = m[i], &out = m[i+1];
        _print_model_parm(fs, in, out);
        
        const char *nm = Model::nname(in.grad_fn);
        fs << nm << std::endl;                /// * one blank line serves
                                              /// * as the sectional break
        IO_DB("\n%2d> %s [%d,%d,%d,%d]\tp=%-2d => out[%d,%d,%d,%d]",
            i, nm, in.N(), in.H(), in.W(), in.C(), in.iparm,
            out.N(), out.H(), out.W(), out.C());
    }
    return 0;
}

__HOST__ int
AIO::_nsave_param(h_ostr &fs, Model &m) {
    auto _dump = [&fs](const char pn, const char *nm, Tensor &t) {
        fs << "\n--- " << pn << "." << nm << std::endl;/// * section marker
        fs.write((char*)t.data, t.numel * sizeof(DU));
    };
    for (U16 i = 1; i < m.numel - 1; i++) {
        Tensor   &in = m[i];                           ///< nth model layer
        t4_layer fn  = in.grad_fn;                     ///< layer function
        const char *nm = Model::nname(fn);             ///< layer name
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
AIO::_nload_model(h_istr &fs, Model &m, char *fname, char *tib) {
    std::string line;
    while (getline(fs, line) && line[0] == '\\') {     /// * TODO: check version
        IO_DB("\n%s", line.c_str());
    }
    if (m.numel > 2) return 0;                         /// * model already loaded

    std::string cmd = line;                            ///< input command
    while (getline(fs, line) && line.length()) {       /// * append layer-by-layer
        IO_DB("\n%s", line.c_str());                   /// * til blank line as break
        cmd.append(" " + line);
    }
    cmd.append(" nn.load ").append(fname);             /// * add parameter reload command
    
    if (cmd.length() >= T4_IBUF_SZ) {                  /// * check buffer size
        *tib = '\0';
        ERROR(" input buffer (T4_IBUF_SZ) overflow!\n");
        return 1;
    }
    strcpy(tib, cmd.c_str());                          /// * fill buffer from model file
    return 0;
}

__HOST__ int
AIO::_nload_param(h_istr &fs, Model &m) {
    auto _read = [this, &fs](const char *pn, const char *nm, Tensor &t) {
        std::string line;                              ///< input string
        IO_DB("\n%s %s[%d,%d,%d,%d] ", nm, pn, t.N(), t.H(), t.W(), t.C());
        while (getline(fs, line) && !line.length());   /// * skip blank lines
        if (line[0]!='-' || line[1]!='-' || line[2]!='-') {
            ERROR(" model format error");
            return 1;
        }
        fs.read((char*)t.data, t.numel * sizeof(DU));  /// * load parameters
        IO_DB("= %ld bytes", fs.gcount());
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

#endif // (T4_DO_OBJ && T4_DO_NN)
