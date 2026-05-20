/**
 * @file
 * @brief Tensorboard Schemauniversal types
 *
 * <pre>Copyright (C) 2026 GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#pragma once
#include "encoder.h"
#include "png.h"

namespace t4::tb::schema {
    
// TensorProto for scalar F32. Canonical encoding matching protobuf serializer:
//   dtype=DT_FLOAT(1), tensor_shape OMITTED (empty=proto3 default),
//   float_val uses packed encoding (wire type 2), NOT non-packed (wire type 5).
U8V scalar_tensor(F32 v) {
    U32 bits;
    std::memcpy(&bits, &v, 4);
    U8 fb[4] = {
        static_cast<U8>( bits        & 0xFF),
        static_cast<U8>((bits >>  8) & 0xFF),
        static_cast<U8>((bits >> 16) & 0xFF),
        static_cast<U8>((bits >> 24) & 0xFF),
    };
        
    Encoder tp;               ///< TensorProto
    tp.s32(1, 1);             /// * dtype = DT_FLOAT
//        tp.raw(2, {});            /// * tensor_shape = empty (scalar), optional
    tp.raw(5, fb, 4);         /// * float_val at field 5 (packed)
        
    return tp.buf();
}

// TensorProto for string (text)
U8V text_tensor(const char *txt) {
    Encoder tp;               ///< TensorProto
    tp.s32(1, 7);             /// * dtype = DT_STRING (7)
//        tp.raw(2, {});            /// * tensor_shape = empty (scalar), optional
    tp.str(8, txt);           /// * string_val
        
    return tp.buf();
}

U8V image_tensor(int w, int h, const U8V& px) {
    auto png = png::raw2png(w, h, px, 3);            ///< convert raw to PNG (RGB)

    Encoder tp;
    tp.s32(1, 7);                                    /// * dtype = DT_STRING

    // TensorShapeProto { dim { size: 3 } } = bytes 12 02 08 03
    static const U8 shape3[] = {0x12, 0x02, 0x08, 0x03};
    tp.raw(2, shape3, 4);                            /// * tensor_shape → field 2
        
    // TB 2.10+ Time Series format: DT_STRING tensor with [w, h, png...]
    // string_val[0] = width as decimal ASCII
    // string_val[1] = height as decimal ASCII  
    // string_val[2] = raw PNG bytes
    tp.str(8, std::to_string(w));                    /// * string_val[0]: width
    tp.str(8, std::to_string(h));                    /// * string_val[1]: height
    tp.raw(8, png.data(), png.size());               /// * string_val[2]: PNG bytes

    return tp.buf();
}
    
/// Plugin metadata – note: metadata goes in Summary.Value field 9
U8V scalar_meta() {
    Encoder pd;                                     ///< SummaryMetadata.PluginData
    pd.str(1, "scalars");                           /// * plugin name
        
    /// content: empty = scalar_plugin{mode=DEFAULT} in proto3
    Encoder meta;                                   ///< SummaryMetadata
    meta.raw(1, pd.buf());
    meta.s32(4, 1);                                 /// * data_class = DATA_CLASS_SCALAR
        
    return meta.buf();
}

U8V text_meta() {
    Encoder pd;                                     ///< SummaryMetadata.PluginData
    pd.str(1, "text");                              /// * plugin_name
    /// empty content for text
        
    Encoder meta;                                   ///< SummaryMetadata
    meta.raw(1, pd.buf());                          /// * PluginData
    meta.s32(4, 2);                                 /// * data_class = DATA_CLASS_TENSOR
        
    return meta.buf();
}

U8V image_meta() {
    /// ImagePluginData content: use FlatBuffers encoding from flatbuf.h
    /// or omit content entirely — TB handles missing content gracefully
    Encoder pd;
    pd.str(1, "images");                            /// * plugin_name
    /// content omitted — TB defaults to max_images=3

    Encoder meta;
    meta.raw(1, pd.buf());
    meta.s32(4, 3);                                 /// * data_class = DATA_CLASS_BLOB_SEQUENCE
    return meta.buf();
}
    
U8V histo_meta() {
    Encoder pd;
    pd.str(1, "histograms");                        /// * plugin_name
    /// empty content for histogram
        
    Encoder meta;
    meta.raw(1, pd.buf());
        
    return meta.buf();
}

} // namespace t4:tb::schema
