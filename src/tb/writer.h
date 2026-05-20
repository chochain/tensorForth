/*
 * @file
 * @brief  —  TensorBoard summary writer (compatible event file writer)
 */
#pragma once
#include <ctime>          // std::time
#include <fstream>        // std::ofstream, std::ios

#include "crc32c.h"       // include types.h
#include "schema.h"       // include encoder.h, png.h
#include "graph.h"        // include encoder.h

namespace t4::tb {

// ─── EventWriter ────────────────────────────────────────────────────────────
class EventWriter {
public:
    explicit EventWriter(const char *path)
        : _file(path, std::ios::binary | std::ios::trunc) {
        if (!_file.is_open())
            throw std::runtime_error((STR("Cannot open event file: ") + path).c_str());
        add_version();
    }
    ~EventWriter() { if (_file.is_open()) _file.close(); }

    void add_version() {
        Encoder event;
        event.f64(1, static_cast<F64>(std::time(nullptr))); // wall_time
        event.s64(2, 0);                                    // step
        event.str(3, "brain.Event:2");                      // file_version

        _write(event.buf());
    }

    // ── Scalar ──────────────────────────────────────────────────────────────
    void add_scalar(const char *tag, F32 v, int step) {
        Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.f32(2, v);                                   // simple_value

        _write(_summary(enc.buf(), step));
    }
   
    void add_scalar_tensor(const char *tag, F32 v, int step) {
        Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.raw(9, schema::scalar_meta());               // metadata → field 9
        enc.raw(8, schema::scalar_tensor(v));            // tensor   → field 8

        _write(_summary(enc.buf(), step));               // for new time-series
    }

    // ── Text (NEW) ──────────────────────────────────────────────────────────
    void add_text(const char *tag, const char *txt, int step) {
        Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.raw(9, schema::text_meta());                 // metadata → field 9
        enc.raw(8, schema::text_tensor(txt));            // tensor   → field 8

        _write(_summary(enc.buf(), step));
    }

    void add_image(
        const char *tag,
        int w, int h, const U8V& px,
        int step) {
        Encoder enc;
        enc.str(1, tag);
        enc.raw(8, schema::image_tensor(w, h, px));      // tensor   → field 8
        enc.raw(9, schema::image_meta());                // metadata → field 9

        _write(_summary(enc.buf(), step));
    }
    
    // ── Histogram ───────────────────────────────────────────────────────────
    void add_histo(
        const char *tag,
        const F64V& values,
        int step,
        int num_buckets = 30) {
        if (values.empty()) return;
        
        F64 vsum = 0, vsumsq = 0;
        for (F64 v : values) { vsum += v; vsumsq += v*v; }
        
        F64  vmin,   vmax;
        F64V limits, counts;
        _buckets(values, num_buckets, vmin, vmax, limits, counts);
        
        Encoder histo;
        histo.f64(1, vmin);
        histo.f64(2, vmax);
        histo.f64(3, static_cast<F64>(values.size()));
        histo.f64(4, vsum);
        histo.f64(5, vsumsq);
        histo.f64(6, limits);
        histo.f64(7, counts);
        
        Encoder enc;
        enc.str(1, tag);                       // tag
        enc.raw(9, schema::histo_meta());      // metadata → field 9
        enc.raw(5, histo.buf());               // histo    → field 5
        
        _write(_summary(enc.buf(), step));
    }

    void add_histo(
        const char *tag,
        const F32  *values,
        const int  numel,
        const int  step,
        const int  n_buckets = 30) {
        F64V dv(numel);
        for (int i = 0; i < numel; i++) dv[i] = (F64)values[i];
        add_histo(tag, dv, step, n_buckets);
    }
    
    // ── Graph ───────────────────────────────────────────────────────────
    void init_graph() {
        _net.clear();
    }
    
    void add_node(const graph::Node& node) {
        _net.push_back(node);
    }

    void add_graph(S64 step=0) {
        Encoder graph;
        for (auto n : _net) {
            graph.raw(1, n.buf());
        }
        
        Encoder event;
        event.s64(2, step);
        event.raw(4, graph.buf());
        
        _write(event.buf());
    }

protected:
    std::ofstream            _file;        ///< output stream
    std::vector<graph::Node> _net;         ///< storage for Graph nodes

    void _write(const U8V& buf) {
        U64 len = buf.size();
        U32 lc = crc32c::mask(crc32c::value(reinterpret_cast<const U8*>(&len), 8));
        U32 dc = crc32c::mask(crc32c::value(buf.data(), buf.size()));
        _file.write(reinterpret_cast<const char*>(&len),       8);
        _file.write(reinterpret_cast<const char*>(&lc),        4);
        _file.write(reinterpret_cast<const char*>(buf.data()), buf.size());
        _file.write(reinterpret_cast<const char*>(&dc),        4);
        
        _file.flush();
    }

    U8V _summary(const U8V& buf, int step) {
        Encoder summary;
        summary.raw(1, buf);                                    // repeated Value
        
        Encoder event;
        event.f64(1, static_cast<F64>(std::time(nullptr)));     // wall_time
        event.s64(2, (S64)step);                                // step
        event.raw(5, summary.buf());                            // summary
        
        return event.buf();
    }

    void _buckets(
        const F64V& values, int nb,
        F64& vmin, F64& vmax, F64V& limits, F64V& counts) {
        vmin = *std::min_element(values.begin(), values.end());
        vmax = *std::max_element(values.begin(), values.end());
        
        if (vmin == vmax) {
            limits.push_back(vmin + 1e-10);
            counts.push_back((F64)values.size());
            return;
        }
        F64 bw = (vmax - vmin) / nb;

        // Add an empty underflow bin so the left edge is represented
        limits.push_back(vmin);        // ← ADD THIS
        counts.push_back(0.0);         // ← ADD THIS

        for (int i = 0; i < nb; i++) {
            limits.push_back(vmin + (i + 1) * bw);
            counts.push_back(0.0);
        }
        limits.back() = vmax + 1e-10;

        for (F64 v : values) {
            int b = std::max(0, std::min(nb - 1, (int)((v - vmin) / bw)));
            counts[b + 1] += 1.0;      // ← offset by 1 to skip underflow bin
        }
    }
};

} // namespace t4::tb
