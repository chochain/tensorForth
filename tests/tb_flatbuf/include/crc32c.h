#pragma once
#include <cstdint>
#include <cstring>

namespace crc32c {

// Crc32c (Castagnoli) lookup table
static uint32_t _kt[256];
static bool     _kt_init = false;

static void init() {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
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

inline uint32_t value(const uint8_t* data, size_t n) {
    if (!_kt_init) init();
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        crc = _kt[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

inline uint32_t value(const char* data, size_t n) {
    return value(reinterpret_cast<const uint8_t*>(data), n);
}

// TensorBoard uses masked CRC:  ((crc >> 15 | crc << 17) + 0xa282ead8u)
inline uint32_t mask(uint32_t crc) {
    return ((crc >> 15) | (crc << 17)) + 0xa282ead8u;
}

} // namespace crc32c
