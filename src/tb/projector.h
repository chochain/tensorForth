/*
 * @file projector.h
 * @brief TensorBoard Embedding Projector support
 *
 * Writes projector_config.pbtxt + TSV tensor/metadata files
 * into the log directory for the TensorBoard Projector plugin.
 */
#pragma once
#include <fstream>
#include <sstream>
//#include <stdexcept>
#include <iterator>
#include "types.h"

namespace t4::tb::embed {

typedef std::string STR;

class Projector {
public:
    struct Embedding {
        STR name;                ///< logical name shown in projector UI
        STR tensor_path;         ///< path to tensors TSV file
        STR metadata_path;       ///< path to metadata TSV file (may be empty)
        USZ dim;                 ///< embedding dimension
    };
    ///
    /// Write a single embedding tensor + optional labels, then register it.
    /// Call flush_config() once after all add_embedding() calls.
    ///
    void add_embedding(
        const char  *logdir,     ///< directory where .tfevents live  (e.g. "tb/run1")
        const char  *tag,        ///< embedding name shown in UI      (e.g. "word_vecs")
        const F32   *data,       ///< flat array: num_points * dim floats, row-major
        USZ         N,           ///< number of embedding vectors
        USZ         HWC,         ///< dimension of each vector
        const char  *meta=NULL)
    {
        STR base    = STR(logdir) + "/" + tag;
        STR t_path  = base + "_tensors.tsv";
        STR m_path  = base + "_metadata.tsv";

        _write_tensors(t_path, data, N, HWC);
        _write_metadata(m_path, tag, N, meta);
        
        _list.push_back({ tag, t_path, m_path, HWC });
    }
    ///
    /// Write projector_config.pbtxt — call once after all add_embedding() calls.
    ///
    void flush_config(const char* logdir) {
        STR path = STR(logdir) + "/projector_config.pbtxt";
        std::ofstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot write: " + path);

        for (const auto& e : _list) {
            f << "embeddings {\n";
            f << "  tensor_name: \"" << e.name << "\"\n";
            f << "  tensor_path: \"" << e.tensor_path << "\"\n";
            if (!e.metadata_path.empty())
                f << "  metadata_path: \"" << e.metadata_path << "\"\n";
            f << "}\n";
        }
    }

    void clean() {
        _list.clear();
    }        

private:
    std::vector<Embedding> _list;

    void _write_tensors(
        const STR& path, const F32* data, USZ N, USZ HWC) {
        std::ofstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot write tensors: " + path);

        for (USZ n = 0; n < N; n++) {
            for (USZ i = 0; i < HWC; i++) {
                if (i) f << '\t';
                f << data[n * HWC + i];
            }
            f << '\n';
        }
    }
    
    void _write_metadata(
        const STR& path, const char *tag, USZ N, const char *meta) {
        std::ofstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot write metadata: " + path);

        // Single-column metadata needs no header; multi-column needs one
        if (meta) {
            STR hdr(meta);
            std::replace(hdr.begin(), hdr.end(), ' ', '\t');
            f << hdr << '\n';
        }
        
        for (int n = 0; n < N; n++) {
            f << tag << '.' << n << '\n';
        }
    }
};

} // namespace t4::tb::embed
