///
/// Unit Test - cueForth ostream
/// 
///> g++ -std=c++14 -Wall -L/home/gnii/devel/catch2 -lcatch test_ostream.cpp && a.out
///
#define  CATCH_CONFIG_MAIN
#include "../../../catch2/catch.hpp"

#undef  __CUDACC__
#define CUEF_OBUF_SIZE  1024

typedef struct { int x; } cudaDim;
cudaDim threadIdx = { 0 }, blockIdx = { 0 };

#include "../src/util.h"
#include "../src/ostream.h"

using namespace cuef;

TEST_CASE("ostream class")
{
    const char *aa[] = {
        "abc def",
        "  abc def",
        "\tabc def",
        "abcdefghi jklmnopqr"
    };
    int na = sizeof(aa)/sizeof(char*);
    SECTION("constructor, length()") {
        char buf1[64];
        ostream v1(buf1, 64);
        REQUIRE(v1.size()==64);
        REQUIRE(v1.tellp()==0);
        
        char buf2[CUEF_OBUF_SIZE];
        ostream v2(buf2);
        REQUIRE(v2.size()==CUEF_OBUF_SIZE);
        REQUIRE(v2.tellp()==0);
    }
    SECTION("<<(U8 c), tellp(), clear()") {
        char buf[CUEF_OBUF_SIZE];
        ostream v1(buf);
        for (int i=0; i<na; i++) {
            int n = strlen(aa[i]);
            for (int j=0; j<n; j++) {
                v1 << (U8)aa[i][j];
                REQUIRE(v1.tellp()==(j+1)*(sizeof(obuf_node)+4));
            }
            v1.clear();
            REQUIRE(v1.tellp()==0);
        }
    }
    GI ii[] = { 123, -456, 7890 };
    int ni = sizeof(ii)/sizeof(GI);
    SECTION("<<(int)") {
        char buf[CUEF_OBUF_SIZE];
        ostream v1(buf);
        for (int i=0; i<ni; i++) {
            v1 << ii[i];
            REQUIRE(v1.tellp()==(i+1)*(sizeof(obuf_node)+4));
        }
    }
    GF ff[] = { 123.0, -456.0, 7890.123 };
    int nf = sizeof(ff)/sizeof(GF);
    SECTION("<<(float)") {
        char buf[CUEF_OBUF_SIZE];
        ostream v1(buf);
        for (int i=0; i<nf; i++) {
            v1 << ff[i];
            REQUIRE(v1.tellp()==(i+1)*(sizeof(obuf_node)+4));
        }
    }
    SECTION("<<(char*)") {
        char buf[CUEF_OBUF_SIZE];
        ostream v1(buf);
        for (int i=0; i<na; i++) {
            v1 << aa[i];
            int n = ALIGN4(strlen(aa[i])+1);
            REQUIRE(v1.tellp()==(sizeof(obuf_node)+n));
            v1.clear();
        }
    }
#ifdef CUEF_USE_STRING
    SECTION("<<(string&)") {
        char buf[CUEF_OBUF_SIZE];
        ostream v1(buf);
        string  s;
        for (int i=0; i<na; i++) {
            s = string(aa[i]);
            v1 << s;
            int n = ALIGN4(strlen(aa[i])+1);
            printf("n=%d", n);
            REQUIRE(v1.tellp()==(sizeof(obuf_node)+n));
            v1.clear();
        }
    }
#endif // CUEF_USE_STRING
}
