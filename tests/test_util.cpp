///
/// Unit Test - cueForth util
/// 
//> g++ -std=c++14 -Wall -L/home/gnii/devel/catch2 -lcatch util.cpp test_util.cpp && a.out
///
#include <sstream>
#define  CATCH_CONFIG_MAIN
#include "../../../catch2/catch.hpp"

#define  __device__ 
#define  __CUDACC__    1
#include "../src/util.h"

TEST_CASE("util functions")
{
    typedef struct {
        const char *str;
        int len;
    } vec;
    vec aa[] = {
        { "",          0 },
        { "1",         1 },
        { "22",        2 },
        { "333",       3 },
        { "999999999", 9 },
        { "12345678901234567890", 20 },
    };
    int na = sizeof(aa)/sizeof(vec);
    SECTION("d_strlen") {
        for (int i=0; i<na; i++) {
            REQUIRE(aa[i].len==d_strlen(aa[i].str, 1));
        }
    }
    SECTION("d_memcpy, d_memcmp") {
        char buf[40];
        for (int i=0; i<na; i++) {
            d_memcpy(buf, aa[i].str, aa[i].len);
            REQUIRE(d_memcmp(buf, aa[i].str, aa[i].len)==0);
        }
    }
    SECTION("d_strcpy, d_strcmp") {
        char buf[40];
        for (int i=0; i<na; i++) {
            d_strcpy(buf, aa[i].str);
            REQUIRE(d_strcmp(buf, aa[i].str)==0);
        }
    }
    typedef struct {
        const char *str;
        int val;
        int len;
    } vec2;
    vec2 bb[] = {
        { "0",          0,     1 },
        { "1",          1,     1 },
        { "12",         12,    2 },
        { "123",        123,   3 },
        { "123456789",  123456789, 9 },
        { "0",          -0,    1 },
        { "-1",         -1,    2 },
        { "-12",        -12,   3 },
        { "-123",       -123,  4 },
        { "-123456789", -123456789, 10 }
    };
    int nb = sizeof(bb)/sizeof(vec2);
    SECTION("d_itoa - decimal") {
        char buf[40];
        for (int i=0; i<nb; i++) {
            int v = d_itoa(bb[i].val, buf, 10);
            REQUIRE(v==bb[i].len);
            REQUIRE(strcmp(buf, bb[i].str)==0);
        }
    }
    vec2 cc[] = {
        { "0",          0,     1 },
        { "1",          1,     1 },
        { "A",          10,    1 },
        { "10",         16,    2 },
        { "123",        0x123, 3 },
        { "FEDC",       0xfedc,4 },
        { "78ABCDEF",   0x78abcdef, 8 }
    };
    int nc = sizeof(cc)/sizeof(vec2);
    SECTION("d_itoa - hex") {
        char buf[40];
        for (int i=0; i<nc; i++) {
            int v = d_itoa(cc[i].val, buf, 16);
            REQUIRE(v==cc[i].len);
            REQUIRE(strcmp(buf, cc[i].str)==0);
        }
    }
    SECTION("d_strtol") {
        char *p;
        for (int i=0; i<nb; i++) {
            int v = (int)d_strtol(bb[i].str, &p, 10);
            REQUIRE(p!=NULL);
            REQUIRE(v==bb[i].val);
        }
        for (int i=0; i<nc; i++) {
            int v = (int)d_strtol(cc[i].str, &p, 16);
            REQUIRE(p!=NULL);
            REQUIRE(v==cc[i].val);
        }
    }
    typedef struct {
        const char *str;
        float       val;
    } vec3;
    vec3 dd[] = {
        { "0.1",        0.1  },
        { "1.9",        1.9  },
        { "20.9",       20.9 },
        { "3000000.1",  3000000.1 },
        { "-0.1",       -0.1  },
        { "-1.9",       -1.9  },
        { "-20.9",      -20.9 },
        { "-3000000.1", -3000000.1 },
        { "-0.1e0",     -0.1  },
        { "1.9e3",      1900.0  },
        { "-20.9E6",    -20900000.0 }
    };
    int nd = sizeof(dd)/sizeof(vec3);
    SECTION("d_strtof") {
        char *p;
        for (int i=0; i<nd; i++) {
            float v = (float)d_strtof(dd[i].str, &p);
            REQUIRE(p!=NULL);
            REQUIRE(v==dd[i].val);
        }
    }
}
