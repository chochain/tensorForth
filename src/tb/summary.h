/** -*- c++ -*-
 * @file
 * @brief Tensorboard class - async IO module implementation
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __TB_SUMMARY_H
#define __TB_SUMMARY_H
#pragma once
#include <iostream>
#include <sstream>                    // ostringstream
#include <sys/stat.h>                 // mkdir, gethostname, getpid (POSIX)
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
    Summary(const char *root = "/tmp/tb", const char *run_id="run1")
        : _root(root), _run_id(run_id), _step(0) {
        mkdir(root, 0755);            /// * create TensorBoard logdir
        init();
    }
    
#if T4_DO_TB
    __HOST__ void init(const char *run_id=NULL);
    __HOST__ void set_step(int step)                     { _step = step; }
    __HOST__ void scalar(const char *tag, F32 v)         { add_scalar(tag, v, _step); }
    __HOST__ void text(const char *tag, const char *txt) { add_text(tag, txt, _step); }
    __HOST__ void image(const char *tag, T4Base &b);
    __HOST__ void tile(const char *tag, T4Base &b, int n_per_row);
    __HOST__ void histo(const char *tag, T4Base &b, int n_bucket);
    __HOST__ void graph(const char *tag, T4Base &b);
#endif // T4_DO_TB

private:
    const char *_root;                       ///< root directory for events
    const char *_run_id;
    int         _step;                       ///< current step of event

    // ─── Path helper ────────────────────────────────────────────────────────────
    // FIX 3: use hostname + PID in filename as TensorBoard 2.x requires
    std::string _logname(std::string &dir, int seq = 0) {
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
