#pragma once

#include "types.h"

namespace crc32c {

// Crc32c (Castagnoli) lookup table
static U32  _kt[256];
static BOOL _kt_init = false;

static void init() {
    for (U32 i = 0; i < 256; ++i) {
        U32 crc = i;
        for (int j = 0; j < 8; ++j) {
            if (crc & 1)
                crc = 0x82F63B78u ^ (crc >> 1);
            else
                crc >>= 1;
        }
        _kt[i] = crc;
    }
    _kt_init = true;
}

inline U32 value(const U8* data, USZ n) {
    if (!_kt_init) init();
    U32 crc = 0xFFFFFFFFu;
    for (USZ i = 0; i < n; ++i) {
        crc = _kt[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

inline U32 value(const char* data, USZ n) {
    return value(reinterpret_cast<const U8*>(data), n);
}

// TensorBoard uses masked CRC:  ((crc >> 15 | crc << 17) + 0xa282ead8u)
inline U32 mask(U32 crc) {
    return ((crc >> 15) | (crc << 17)) + 0xa282ead8u;
}

} // namespace crc32c
