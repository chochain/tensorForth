/** -*- c++ -*-
 * @file
 * @brief AIO class - async IO module implementation
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <cstdio>        // printf
#include <iostream>      // cin, cout
#include <iomanip>       // setbase, setprecision
#include "dataset.h"     // in ../mmu
#include "model.h"       // in ../mmu
#include "loader.h"      // in ../ldr (include corpus.h)
#include "aio.h"
///
/// AIO takes managed memory blocks as input and output buffers
/// which can be access by both device and host
///
using namespace std;

#define NEXTNODE(n) ((obuf_node*)((char*)&node->data[0] + node->sz))

__HOST__ int
AIO::readline(std::istream &fin) {
    _istr->clear();
    char *tib = _istr->rdbuf();
    fin.getline(tib, T4_IBUF_SZ, '\n');
    return strlen(tib);
}

__HOST__ obuf_node*
AIO::process_node(std::ostream &fout, obuf_node *node) {
    cudaDeviceSynchronize();        /// * make sure data is completely written
    
    char *v = (char*)node->data;    ///< fetch payload in buffered print node
    switch (node->gt) {
    case GT_INT:   fout << (*(I32*)v); break;
    case GT_FLOAT: fout << (*(F32*)v); break;
    case GT_STR:   fout << v;          break;
    case GT_FMT:   {
        obuf_fmt *f = (obuf_fmt*)v;
        //printf("FMT: b=%d, w=%d, p=%d, f='%c'\n", f->base, f->width, f->prec, f->fill);
        fout << std::setbase(_radix = f->base)
             << std::setw(f->width)
             << std::setprecision(f->prec ? f->prec : -1)
             << std::setfill((char)f->fill);
    } break;
    case GT_OBJ: _print_obj(fout, *(DU*)v); break;
    case GT_OPX: {
        _opx *o = (_opx*)v;
        // printf("OP=%d a=%d, n=0x%08x=%f\n", o->op, o->a, DU2X(o->n), o->n);
        switch (o->op) {
        case OP_WORDS: _mmu->words(fout);                               break;
        case OP_SEE:   _mmu->see(fout, (IU)o->a);                       break;
        case OP_DUMP:  _mmu->mem_dump(fout, (IU)o->a, (IU)o->n);        break;
        case OP_SS:    _mmu->ss_dump(fout, (IU)node->id, o->a, _radix); break;
        case OP_DATA:
            node = NEXTNODE(node);                          ///< get dataset repo name
            _fetch(o->n, (bool)o->a, (char*)node->data);    /// * fetch first batch
            break;
        case OP_FETCH: _fetch(o->n, (bool)o->a); break;     /// * fetch/rewind dataset batch
        case OP_NSAVE:
            node = NEXTNODE(node);                          ///< get dataset repo name
            _nsave(o->n, (bool)o->a, (char*)node->data);
            break;
        case OP_NLOAD:
            node = NEXTNODE(node);
            _nload(o->n, (bool)o->a, (char*)node->data);
            break;
        }
    } break;
    default: fout << "print type not supported: " << (int)node->gt; break;
    }
    return NEXTNODE(node);
}

__HOST__ void
AIO::flush(std::ostream &fout) {
    obuf_node *node = (obuf_node*)_ostr->rdbuf();
    while (node->gt != GT_EMPTY) {          // 0
        node = process_node(fout, node);
    }
    _ostr->clear();
}
///
/// private methods
///
#if T4_ENABLE_OBJ
__HOST__ void
AIO::_print_obj(std::ostream &fout, DU v) {
    T4Base &b = _mmu->du2obj(v);
    switch (b.ttype) {
    case T4_VIEW:
    case T4_TENSOR:
    case T4_DATASET: _print_tensor(fout, v); break;
    case T4_MODEL:   _print_model(fout, v);  break;
    }
}
__HOST__ void
AIO::_print_vec(std::ostream &fout, DU *d, int mj, int rj, int c) {
    fout << setprecision(_prec) << "{";                 /// set precision
    for (int j=0; j<rj; j++) {
        DU *dx = &d[j * c];
        for (int k=0; k < c; k++) {
            fout << (k>0 ? "_" : " ") << *dx++;
        }
    }
    int x = mj - rj;
    if (x > rj) fout << " ...";
    for (int j=(x > rj ? x : rj); j<mj; j++) {
        DU *dx = &d[j * c];
        for (int k=0; k < c; k++) {
            fout << (k>0 ? "_" : " ") << *dx++;
        }
    }
    fout << " }";
}
__HOST__ void
AIO::_print_mat(std::ostream &fout, DU *d, int mi, int mj, int ri, int rj, int c) {
    fout.flags(ios::showpos | ios::right | ios::fixed); /// enforce +- sign
    bool full = (mi * mj) <= _thres;
    int  x    = full ? mj : rj;
    DU   *d0  = d;
    for (int i=0, i1=1; i<ri; i++, i1++, d0+=(mj * c)) {
        _print_vec(fout, d0, mj, x, c);
        fout << (i1==mi ? "" : "\n\t");
    }
    int y = full ? ri : mi - ri;
    if (y > ri) fout << "...\n\t";
    else y = ri;
    DU *d1 = (d + y * mj * c);
    for (int i=y, i1=i+1; i<mi; i++, i1++, d1+=(mj * c)) {
        _print_vec(fout, d1, mj, x, c);
        fout << (i1==mi ? "" : "\n\t");
    }
}
__HOST__ void
AIO::_print_tensor(std::ostream &fout, DU v) {
    auto range = [this](int n) { return (n < _edge) ? n : _edge; };

    Tensor &t = (Tensor&)_mmu->du2obj(v);
    DU     *d = t.data;                     /// * short hand
    WARN("aio#print_tensor::T[%x]=%p data=%p\n", DU2X(v), &t, d);

    ios::fmtflags fmt0 = fout.flags();
    fout << setprecision(-1);               /// * standard format
    switch (t.rank) {
    case 1: {
        fout << "vector[" << t.numel << "] = ";
        int ri = (t.numel < _thres) ? t.numel : range(t.numel);
        _print_vec(fout, d, t.numel, ri, 1);
    } break;
    case 2: {
        int mi = t.H(), mj = t.W(), ri = range(mi),  rj = range(mj);
        fout << "matrix[" << mi << "," << mj << "] = {\n\t";
        _print_mat(fout, d, mi, mj, ri, rj, 1);
        fout << " }";
    } break;
    case 4: {
        int n  = t.N(), mi = t.H(), mj = t.W(), mc = t.C();
        int ri = range(mi), rj = range(mj);
        int pg = mi * mj * mc;
        fout << "tensor["
             << n << "," << mi << "," << mj << "," << mc
             << "] = {\n\t";
        for (int i = 0; i < n; i++, d += pg) {
            if (mj==1) _print_mat(fout, d, mj, mi, rj, ri, mc);
            else       _print_mat(fout, d, mi, mj, ri, rj, mc);
            fout << ((i+1) < n ? "\n\t" : "");
        }
        fout << " }";
    } break;
    case 5: {
        fout << "tensor[" << t.parm << "]["
             << t.N() << "," << t.H() << "," << t.W() << "," << t.C()
             << "] = {...}";
    } break;        
    default: fout << "tensor rank=" << t.rank << " not supported";
    }
    fout << "\n";
    fout.flags(fmt0);
}
__HOST__ void
AIO::_print_model(std::ostream &fout, DU v) {
    auto tinfo = [this, &fout](Tensor &t, int i, int fn) { ///> layer info
        fout << "[" << std::setw(3) << i << "] "
             << Model::nname(fn) << ":";
        _mmu->to_s(fout, t);
        int sz = 0;
        for (int n = 0; n < 4; n++) {
            sz += t.grad[n] ? t.grad[n]->numel : 0;
        }
        fout << ", #param=" << sz;
    };
    auto finfo = [this, &fout](Tensor **g) {
        for (int i=0; g[i] && i < 2; i++) {
            fout << " "; _mmu->to_s(fout, *g[i]);
        }
    };
    Model &m = (Model&)_mmu->du2obj(v);
    int   sz = m.numel;
    if (!m.is_model()) return;
    
    fout << "NN model[" << sz-1 << "/" << m.slots() << "]" << endl;
    for (int i = 1; i < sz; i++) {  /// skip root[0]
        Tensor &t = m[i];
        tinfo(t, i, (i==(sz-1)) ? 0 : t.grad_fn);
        if (_trace && t.grad_fn != L_NONE) finfo(t.grad);
        fout << endl;
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
///
__HOST__ int
AIO::_fetch(DU top, bool more, char *ds_name) {
    Dataset &ds = (Dataset&)_mmu->du2obj(top);    ///< dataset ref
    U32     dsx = DU2X(top);                      ///< dataset mnemonic
    if (!ds.is_dataset()) {                       /// * indeed a dataset?
        ERROR("mmu#load TOS is not a dataset\n");
        return -1;
    }
    ///
    /// search cache for top <=> dataset pair
    ///
    TRACE1("\nAIO::%s dataset (id=%x) =>",
           ds_name ? ds_name : (more ? "fetch" : "rewind"), dsx);
    Corpus *cp = Loader::get(dsx, ds_name);      ///< Corpus/Dataset provider
    if (!cp) {
        ERROR(" dataset not found\n"); return -1;
    }
    if (more==0 && ds.batch_id >= 0) {            /// rewind dataset
        cp->rewind();
        ds.batch_id = ds.done = 0;
    }
    else if ((ds.done=cp->eof)) {                /// * dataset exhausted?
        TRACE1(" completed, no more data.\n"); return 0;
    }
    ///
    /// init and load a batch of data points
    ///
    int batch_sz = ds.N();                        ///< dataset batch size
    int bid = ds.batch_id < 0 ? 0 : ds.batch_id;  ///< batch_id to fetch
    if (!cp->fetch(bid, batch_sz)) {              /// * fetch a batch from Corpus
        ERROR("fetch failed\n");  return -2;
    }
    if (ds.batch_id < 0) {
        ds.reshape(batch_sz, cp->H, cp->W, cp->C);
        ds.batch_id = 1;                          /// * ready for next batch
    }
    else ds.batch_id++;
    ///
    /// transfer host into device memory
    /// if needed, allocate Dataset device (managed) memory blocks
    ///
    if (!cp->eof) ds.load_batch(cp->data, cp->label);
    TRACE1("batch[%d] %d record(s) loaded\n", ds.batch_id - 1, batch_sz);
    
    return 0;
}
///
/// NN model persistence (i.e. serialization) methods
///
#include <fstream>
__HOST__ int
AIO::_nsave(DU top, U16 vid, char* fname) {
    printf("\n%d|AIO::save model to '%s' =>", vid, fname);
    Model &m = (Model&)_mmu->du2obj(top);
    ofstream fout(fname, ios_base::binary);     ///< open an output file
    if (!fout.is_open()) {
        ERROR(" failed to open for output\n");
        return 1;
    }
    fout << "\\ " << T4_APP_NAME << " model\n\\ version v" << T4_MAJOR_VER << "." << T4_MINOR_VER << "\n";
    _nsave_model(fout, m);                       /// * blank line as section break
    _nsave_param(fout, m);
    fout << "\n---" << endl;
    fout.close();
    printf(" completed\n");
    return 0;
}

__HOST__ int
AIO::_nload(DU top, U16 vid, char* fname) {
    printf("\n%d|AIO::load '%s' ", vid, fname);
    Model &m = (Model&)_mmu->du2obj(top);
    ifstream fin(fname, ios_base::binary);           ///< open an input file
    if (!fin.is_open()) {
        ERROR("=> failed to open for input\n");
        return 1;
    }
    int err = 0;
    if (m.numel <= 2) {
        printf("NN model");
        err = _nload_model(fin, m, fname);            /// * load model layers
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
AIO::_nsave_model(std::ostream &fout, Model &m) {
    for (U16 i = 1; i < m.numel - 1; i++) {
        Tensor &in = m[i], &out = m[i+1];
        t4_layer fn = in.grad_fn;              ///< layer function
        DU       p  = 0.001 * in.parm;         ///< layer parameter
        switch(fn) {
        case L_CONV:   fout << p << " " << out.C() << " "; break;
        case L_LINEAR: fout << p << " " << out.H() << " "; break;
        case L_AVGPOOL:
        case L_MAXPOOL:
        case L_MINPOOL: fout << in.parm << " ";            break;
        case L_DROPOUT: fout << p << " ";                  break;
        case L_USAMPLE: fout << (in.parm&0xff) << "[" << (in.parm>>8) << "] "; break;
        case L_BATCHNM: fout << p << " ";                  break;
        default: break;
        }
        const char *nm = Model::nname(fn);
        fout << nm << endl;                              /// * one blank line serves
                                                         /// * as the sectional break
        printf("\n%2d> %s [%d,%d,%d,%d]\tp=%-2d => out[%d,%d,%d,%d]",
            i, nm, in.N(), in.H(), in.W(), in.C(), in.parm,
            out.N(), out.H(), out.W(), out.C());
        
    }
    return 0;
}

__HOST__ int
AIO::_nsave_param(std::ostream &fout, Model &m) {
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
        default: break;
        }
    }
    return 0;
}

__HOST__ int
AIO::_nload_model(std::istream &fin, Model &m, char *fname) {
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
AIO::_nload_param(std::istream &fin, Model &m) {
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
        default: break;
        }
    }
    return 0;
}
#endif // T4_ENABLE_OBJ
