#pragma once
#include <cstdint>
#include <cstring>

namespace crc32c {

// CRC32C (Castagnoli) lookup table
static uint32_t kTable[256];
static bool     kTableInit = false;

static void InitTable() {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int j = 0; j < 8; ++j) {
            if (crc & 1)
                crc = 0x82F63B78u ^ (crc >> 1);
            else
                crc >>= 1;
        }
        kTable[i] = crc;
    }
    kTableInit = true;
}

inline uint32_t Value(const uint8_t* data, size_t n) {
    if (!kTableInit) InitTable();
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        crc = kTable[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

inline uint32_t Value(const char* data, size_t n) {
    return Value(reinterpret_cast<const uint8_t*>(data), n);
}

// TensorBoard uses masked CRC:  ((crc >> 15 | crc << 17) + 0xa282ead8u)
inline uint32_t Masked(uint32_t crc) {
    return ((crc >> 15) | (crc << 17)) + 0xa282ead8u;
}

} // namespace crc32c
