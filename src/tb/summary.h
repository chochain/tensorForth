/** -*- c++ -*-
 * @file
 * @brief Tensorboard class - async IO module implementation
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __TB_SUMMARY_H
#define __TB_SUMMARY_H
#pragma once
#include <sstream>                    // ostringstream
#include <sys/stat.h>                 // mkdir (POSIX)
#include "writer.h"

namespace t4::nn { class Model;  }    /// forward declare
namespace t4::mu { class Tensor; }
namespace t4::tb {

class Summary : public EventWriter {
#if T4_DO_OBJ
    using Tensor  = mu::Tensor;       ///< aliases
#if T4_DO_NN    
    using Model   = nn::Model;
#endif // DO_NN
#endif // DO_OBJ
    
public:
    Summary(const char *root = "/tmp/tb") : _root(root), _step(0) {
        mkdir(root, 0755);                    /// * create TensorBoard logdir
    }
    
#if T4_DO_TB
    __HOST__ void init(const char *run_id) {
        const char *rundir = (std::string(_root) + "/" + run_id).c_str();
        mkdir(rundir, 0755);                  /// * create Event/Run subdir
        setup(_logname(rundir).c_str());
    }
    __HOST__ void set_step(int step)                     { _step = step; }
    __HOST__ void scalar(const char *tag, F32 v)         { add_scalar(tag, v, _step); }
    __HOST__ void text(const char *tag, const char *txt) { add_text(tag, txt, _step); }
    __HOST__ void image(const char *tag, Tensor &t);
    __HOST__ void tile(const char *tag, Tensor &t, int n_per_row);
    __HOST__ void histo(const char *tag, Tensor &t, int n_bucket);
    __HOST__ void graph(const char *tag, Model &m);
#endif // T4_DO_TB

private:
    const char *_root;                       ///< root directory for events
    int         _step;                       ///< current step of event

    // ─── Path helper ────────────────────────────────────────────────────────────
    // FIX 3: use hostname + PID in filename as TensorBoard 2.x requires
    std::string _logname(const char *dir, int seq = 0) {
        char hostname[256] = "localhost";
        gethostname(hostname, sizeof(hostname));
        hostname[sizeof(hostname)-1] = '\0';
        
        for (char* p = hostname; *p; ++p)
            if (*p == '/' || *p == '\\' || *p == ':') *p = '_';
        
        std::ostringstream ss;
        ss << dir << "/events.out.tfevents."
           << static_cast<long>(std::time(nullptr)) << "."
           << hostname << "."
           << static_cast<int>(getpid()) << "." << seq;
        
        return ss.str();
    }
};

} // namespace t4::tb

#endif // __TB_SUMMARY_H
