///
/// Unit Test - cueForth istream
/// 
///> g++ -std=c++14 -Wall -L/home/gnii/devel/catch2 -lcatch test_istream.cpp && a.out
///
#define  CATCH_CONFIG_MAIN
#include "../../../catch2/catch.hpp"

#define CUEF_OBUF_SIZE 1024
#define __GPU__
#define __KERN__
#define __HOST__
#define __INLINE__
typedef int   GI;
typedef float GF;

#include "../src/sstream.h"

using namespace cuef;

TEST_CASE("istream class")
{
    const char *aa[] = {
        "abc def",
        "  abc def",
        "\tabc def",
        "abcdefghi jklmnopqr"
    };
    int nn = sizeof(aa)/sizeof(char*);
    const char *rst[nn][2] = {
        { "abc", "def" },
        { "abc", "def" },
        { "abc", "def" },
        { "abcdefghi", "jklmnopqr" }
    };
    SECTION("str(char*,sz), str(string&), size(), gcount()") {
        istream v1;
        istream v2;
        for (int i=0; i<nn; i++) {
            v1.str(aa[i]);
            string s2(aa[i]);
            v2.str(s2);
            REQUIRE(v1.size()==(int)strlen(aa[i]));
            REQUIRE(v2.gcount()==(int)strlen(aa[i]));
        }
    }
    SECTION("getline(char*)") {
        char buf[20];
        istream v1;
        for (int i=0; i<nn; i++) {
            v1.str(aa[i]);
            v1.getline(buf);
            REQUIRE(strcmp(buf, rst[i][0])==0);
            v1.getline(buf);
            REQUIRE(strcmp(buf, rst[i][1])==0);
        }
    }
    SECTION("getline(string&)") {
        string s;
        istream v1;
        for (int i=0; i<nn; i++) {
            v1.str(aa[i]);
            v1.getline(s);
            REQUIRE(strcmp(s.c_str(), rst[i][0])==0);
            v1.getline(s);
            REQUIRE(strcmp(s.c_str(), rst[i][1])==0);
        }
    }
    SECTION(">>(char*), >>(string&)") {
        char buf[20];
        string s;
        istream v1;
        for (int i=0; i<nn; i++) {
            v1.str(aa[i]);
            int sz = v1 >> buf;
            REQUIRE(sz==(int)strlen(rst[i][0]));
            REQUIRE(strcmp(buf, rst[i][0])==0);
            v1 >> s;
            REQUIRE(sz==(int)strlen(rst[i][1]));
            REQUIRE(strcmp(s.c_str(), rst[i][1])==0);
        }
    }
}
