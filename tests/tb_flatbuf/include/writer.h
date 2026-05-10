/*
 @file
 @brief  —  TensorBoard/tbparse compatible event file writer
 @note   —  see schema.h for details
 */
#pragma once

#include "types.h"
#include "crc32c.h"
#include "schema.h"
#include "png.h"
#include "encoder.h"

#include <ctime>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <map>

namespace tensorboard {
// ─── Path helper ────────────────────────────────────────────────────────────
inline STR logdir(const STR& dir, int seq = 0) {
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

void _dump(const U8V& buf, const char *hdr, const char *pfx="") {
    int sz = (int)buf.size();
    printf("%s%s len=%d(%x)\n", pfx, hdr, sz, sz);
    for (int i=0; i < sz; i+=16) {
        printf("%s%04x:", pfx, i);
        for (int j=0; j<16; j++) {
            U8 c = (i+j) < sz ? buf.data()[i+j] : 0;
            printf(" %02x", c);
        }
        printf("  ");
        for (int j=0; j < 16; j++) {   // print and advance to next byte
            U8 c = ((i+j) < sz ? buf.data()[i+j] : 0) & 0x7f;
            printf("%c", (char)((c==0x7f||c<0x20) ? '_' : c));
        }
        printf("\n");
    }
}

// ─── EventWriter ────────────────────────────────────────────────────────────
class EventWriter {
public:
    explicit EventWriter(const STR& path)
        : _file(path, std::ios::binary | std::ios::trunc) {
        if (!_file.is_open())
            throw std::runtime_error("Cannot open event file: " + path);
        add_version();
    }
    ~EventWriter() { if (_file.is_open()) _file.close(); }

    void add_version() {
        proto::Encoder event;
        event.f64(1, static_cast<F64>(std::time(nullptr))); // wall_time
        event.s64(2, 0);                                    // step
        event.str(3, "brain.Event:2");                      // file_version

        _write(event.buf());
    }

    // ── Scalar ──────────────────────────────────────────────────────────────
    void add_scalar(const STR& tag, F32 v, S64 step) {
        proto::Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.f32(2, v);                                   // simple_value

        _write(_summary(enc.buf(), step));
    }
   
    void add_scalar_tensor(const STR& tag, F32 v, S64 step) {
        proto::Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.raw(9, schema::scalar_meta());               // metadata → field 9
        enc.raw(8, schema::scalar_tensor(v));            // tensor   → field 8

        _write(_summary(enc.buf(), step));               // for new time-series
    }

    // ── Text (NEW) ──────────────────────────────────────────────────────────
    void add_text(const STR& tag, const STR& txt, S64 step) {
        proto::Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.raw(9, schema::text_meta());                 // metadata → field 9
        enc.raw(8, schema::text_tensor(txt));            // tensor   → field 8

        _write(_summary(enc.buf(), step));
    }

    // ── Image (RGB row-major, 3 bytes/pixel) ────────────────────────────────
    void add_image(
        const STR& tag,
        int width,
        int height,
        const U8V& pixels_rgb,
        S64 step) {
        auto png = png::raw2png(width, height, pixels_rgb, 3);
        
        proto::Encoder img;
        img.s32(1, height);
        img.s32(2, width);
        img.s32(3, 3);                                   // colorspace = RGB
        img.raw(4, png.data(), png.size());              // encoded_image_string
        
        proto::Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.raw(9, schema::image_meta());                // metadata → field 9
        enc.raw(4, img.buf());                           // image    → field 4
        
        _write(_summary(enc.buf(), step));
    }

    // ── Histogram ───────────────────────────────────────────────────────────
    void add_histo(
        const STR& tag,
        const F64V& values,
        S64 step,
        int num_buckets = 30) {
        if (values.empty()) return;
        
        F64 vmin = *std::min_element(values.begin(), values.end());
        F64 vmax = *std::max_element(values.begin(), values.end());
        F64 vsum = 0, vsumsq = 0;
        for (F64 v : values) { vsum += v; vsumsq += v*v; }
        
        F64V limits, counts;
        _buckets(vmin, vmax, values, num_buckets, limits, counts);
        
        proto::Encoder histo;
        histo.f64(1, vmin);
        histo.f64(2, vmax);
        histo.f64(3, static_cast<F64>(values.size()));
        histo.f64(4, vsum);
        histo.f64(5, vsumsq);
        histo.f64(6, limits);
        histo.f64(7, counts);
        
        proto::Encoder enc;
        enc.str(1, tag);                       // tag
        enc.raw(9, schema::histo_meta());      // metadata → field 9
        enc.raw(5, histo.buf());               // histo    → field 5
        
        _write(_summary(enc.buf(), step));
    }

    void add_histo(
        const STR& tag,
        const F32V& values,
        S64 step,
        int num_buckets = 30) {
        F64V dv(values.begin(), values.end());
        add_histo(tag, dv, step, num_buckets);
    }

protected:
    std::ofstream _file;

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

    U8V _summary(const U8V& buf, S64 step) {
        proto::Encoder summary;
        summary.raw(1, buf);                                    // repeated Value
        
        proto::Encoder event;
        event.f64(1, static_cast<F64>(std::time(nullptr)));     // wall_time
        event.s64(2, step);                                     // step
        event.raw(5, summary.buf());                            // summary
        
        return event.buf();
    }

    void _buckets(
        F64 vmin, F64 vmax, const F64V& values,
        int nb, F64V& limits, F64V& counts) {
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

} // namespace tensorboard
