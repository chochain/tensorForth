/** -*- c++ -*-
 * @file
 * @brief System class - tensorForth System interface implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iostream>
#include <chrono>
#include <thread>
#include "mu/tensor.h"
#include "mu/dataset.h"
#include "nn/model.h"
#include "sys.h"                     /// include mmu/tensor.h

namespace t4 {
using io::AIO;
using mu::MMU;

System  *_sys = NULL;                ///< singleton controller on host
///
/// Forth Virtual Machine operational macros to reduce verbosity
///
__HOST__
System::System(h_istr &i, h_ostr &o, int khz, int verbo)
    : fin(i), fout(o),
      _istr(new io::Istream()), _ostr(new io::Ostream()),
      _khz(khz), _trace(verbo) {
    mu = MMU::get_mmu();             /// * instantiate memory controller
    io = AIO::get_io(&_trace);       /// * instantiate async IO controler
    db = Debug::get_db(o);           /// * tracing instrumentation
    tb = NULL;
    ///
    ///> setup randomizer
    ///
    k_rand_init<<<1, T4_RAND_SZ>>>(time(NULL));  /// serialized randomizer
    GPU_CHK();

    INFO("\\ System OK\n");
}

__HOST__
System::~System() {
//    cudaDeviceSynchronize();        /// * sync before freeing everything
    
    AIO::free_io();
    Debug::free_db();
    MMU::free_mmu();
    INFO("\\ System: instance freed\n");
}

__HOST__ System*
System::get_sys(h_istr &i, h_ostr &o, int khz, int verbo) {
    if (!_sys) _sys = new System(i, o, khz, verbo);
    return _sys;
}
__HOST__ System *System::get_sys()  { return _sys; }
__HOST__ void    System::free_sys() { if (_sys) delete _sys; }
///
///@name cross platform timing support
///
__HOST__ DU
System::clock() {
    return 1.0f * std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

__HOST__ void
System::delay(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
///
///@name Randomizer
///
__HOST__ void
System::rand(DU d, rand_opt o) {
#if T4_DO_OBJ
    mu::Tensor &t = (mu::Tensor&)MMU::get_mmu()->du2obj(d);
    rand(t.data, t.numel, o);
#else  // !T4_DO_OBJ
    ERROR("n/a");
#endif // T4_DO_OBJ    
}

__HOST__ void
System::rand(DU *d, U64 sz, rand_opt o, DU bias, DU scale) {
    /// rand states are dependent, cannot run parallel with multi-blocks
    k_rand<<<1, T4_RAND_SZ>>>(d, sz, bias, scale, o);
    GPU_CHK();
}
///@}
///@name TensorBoard support
///@{
__HOST__ void
System::setup_tb(const char *tb_logdir, const char *tb_run_id) {
    tb = new tb::Summary(tb_logdir, tb_run_id);     /// * TensorBoard streamer (_logdir=/u01/tb/)
}
///@}
///@name event loop handler
///@{
#include <iostream>
#include <string>
#define NEXT_EVENT(ev) ((io::event*)((char*)&(ev)->data[0] + (ev)->sz))

__HOST__ int
System::readline(int hold) {                 ///< feed a line into device input stream
    if (hold) return 1;                      ///< do not clear input buffer
    _istr->clear();                          /// * clear device inpugt stream
    char *tib = _istr->rdbuf();              ///< set device input buffer
    fin.getline(tib, T4_IBUF_SZ, '\n');      /// * feed input from host to device
    return !fin.eof();                       /// * more to read?
}

__HOST__ void
System::flush() {
    io::event *e = (io::event*)_ostr->rdbuf(); /// * managed mem read, auto sync
    while (e->gt != GT_EMPTY) {              /// * walk linked-list
        e = e->gt == GT_OPX                  /// * process composit or simple event
            ? _process_opx(e)
            : (e->gt == GT_TBX) ? _process_tb(e) : _process_event(e);
    }
    fout << std::flush;
    _ostr->clear();
}

__HOST__ io::event*
System::_process_event(io::event *ev) {      ///< process simple IO requests
    void *vp = (void*)ev->data;              ///< fetch payload in buffered print node
    DEBUG("System::process(gt=%x) {\n", ev->gt);
    switch (ev->gt) {
    case GT_INT:
    case GT_U32:
    case GT_FLOAT:
    case GT_STR:
    case GT_FMT: db->print(vp, ev->gt);    break;
    case GT_OBJ: db->print_obj(*(DU*)vp);  break;
    default: ERROR("event type not supported: %d\n", (int)ev->gt); break;
    }
    DEBUG("} System::process(gt=%x)\n", ev->gt);
    return NEXT_EVENT(ev);
}

__HOST__ io::event*
System::_process_opx(io::event *ev) {        ///< process composit IO types
    void *vp = (void*)ev->data;              ///< fetch payload in buffered print node
    io::_opx o = *((io::_opx*)vp);           ///< capture a hardcopy
    DEBUG("  _opx(OP=%d, m=%d, i=%d, n=0x%08x=%g)\n", o.op, o.m, o.i, DU2X(o.n), o.n);
    switch (o.op) {
    case OP_FLUSH: fout << std::flush;           break;
    case OP_DICT:  db->dict_dump();              break;
    case OP_WORDS: db->words();                  break;
    case OP_SEE:   db->see(o.i, o.m);            break;
    case OP_DUMP:  db->mem_dump(UINT(o.n), o.i); break;
    case OP_SS:    db->ss_dump(o.n, o.i, o.m);   break;
#if T4_DO_OBJ // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    case OP_T2PNG:
    case OP_TSAVE: {
        mu::Tensor &t = (mu::Tensor&)mu->du2obj(o.n);
        if (t.is_tensor() || t.is_dataset()) {
            ev = NEXT_EVENT(ev);
            char *fn = (char*)ev->data;                     ///> filename
            OP_T2PNG
                ? io->t2png(t, fn)
                : io->tsave(t, fn, o.m);                    /// * persist for NumPy
        }
        else ERROR("%x is not a tensor\n", DU2X(o.n));
    } break;
#if T4_DO_NN  //==========================================================
    case OP_DATA: {
        mu::Dataset &ds = (mu::Dataset&)mu->du2obj(o.n);
        if (ds.is_dataset()) {                              /// * indeed a dataset?
            ev = NEXT_EVENT(ev);                            ///< get dataset repo name
            char *ds_nm = (char*)ev->data;                  /// * dataset name
            ds.fetch(ds_nm, 0, _trace);                     /// * fetch first batch
        }
        else ERROR("%x is not a dataset\n", DU2X(o.n));
    } break;
    case OP_NORM: {
        mu::Dataset &ds = (mu::Dataset&)mu->du2obj(o.n);
        if (ds.is_dataset()) {
            ev = NEXT_EVENT(ev);                            ///< get dataset repo name
            io::_opx x = *((io::_opx*)ev->data);            ///< capture a hardcopy
            INFO("  OP_NORM(mean=%d, scale=%g)\n", x.i, x.n);
            ds.normalize(I2D((int)x.i), x.n);               /// * fetch first batch
            ds.rewind(_trace);                              /// * rewind/load dataset
        }
        else ERROR("%x is not a dataset\n", DU2X(o.n));
    } break;  
    case OP_FETCH: {
        mu::Dataset &ds = (mu::Dataset&)mu->du2obj(o.n);
        if (ds.is_dataset()) {
            ds.fetch(NULL, o.m, _trace);                    /// * fetch/rewind dataset batch
        }
        else ERROR("%x is not a dataset\n", DU2X(o.n));
    } break;  
    case OP_NSAVE: {
        nn::Model &m = (nn::Model&)mu->du2obj(o.n);
        if (m.is_model()) {
            ev = NEXT_EVENT(ev);                            ///< get dataset repo name
            char *fn = (char*)ev->data;                     ///< filename
            io->nsave(m, fn, o.m);                          /// * o->m FAM mode
        }
        else ERROR("%x is not a model\n", DU2X(o.n));
    } break;
    case OP_NLOAD: {
        nn::Model &m = (nn::Model&)mu->du2obj(o.n);
        if (m.is_model()) {
            ev = NEXT_EVENT(ev);
            _istr->clear();
            char *fn = (char*)ev->data;                    ///< filename
            io->nload(m, fn, o.m, _istr->rdbuf());         /// * fetch into rdbuf
        }
        else ERROR("%x is not a model\n", DU2X(o.n));
    } break;
    }
#endif // T4_DO_NN =======================================================
#endif // T4_DO_OBJ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return NEXT_EVENT(ev);
}

__HOST__ io::event*
System::_process_tb(io::event *ev) {              ///< process TensorBoard ops
#if T4_DO_TB  //==========================================================
    void *vp   = (void*)ev->data;                 ///< fetch payload in buffered print node
    io::_tbx x = *(io::_tbx*)vp;                  ///< make a hardcopy

    if (!tb || _trace) INFO("  sys#tbx(op=%d, n=%g, i=%d", x.op, x.n, x.i);
    ev = NEXT_EVENT(ev);
    
    /// * opcodes withut tag
    if (x.op == TB_STEP || x.op == TB_GRAPH) {
        if (tb) {
            switch (x.op) {
            case TB_STEP:  tb->set_step(x.i);             break;
            case TB_GRAPH: tb->graph(mu->du2obj(x.n));    break;
            }
            if (_trace) INFO(")\n");
        }
        else INFO("), check TensorBoard param -tlogdir -rrun_id\n");
        return ev;
    }
    /// * opcode with tags
    const char *tag = (const char*)ev->data;      ///< retrieve tag for Tensorboard
    if (!tb || _trace) INFO(", tag=%s)\n", tag);
    if (!tb) {
        if (x.op==TB_TEXT) ev = NEXT_EVENT(ev);
        return NEXT_EVENT(ev);
    }
    switch (x.op) {
    case TB_INIT:   tb->init(tag);   /* tag as logname */ break;
    case TB_SCALAR: tb->scalar(tag, x.n);                 break;
    case TB_TEXT: {
        ev = NEXT_EVENT(ev);
        const char *txt = (const char*)ev->data;  ///< get a hardcopy
        if (_trace) INFO("    txt=%s\n", txt);
        tb->text(tag, txt);
    } break;
    case TB_IMAGE: tb->image(tag, mu->du2obj(x.n));       break;
    case TB_TILE:  tb->tile(tag,  mu->du2obj(x.n), x.i);  break;
    case TB_HISTO: tb->histo(tag, mu->du2obj(x.n), x.i);  break;
    case TB_EMBED: tb->embed(tag, mu->du2obj(x.n));       break;
    }
#endif // T4_DO_TB        
    return NEXT_EVENT(ev);
}
///@}

} // namespace t4


