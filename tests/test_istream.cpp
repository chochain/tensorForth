///
/// Unit Test - cueForth istream
/// 
///> g++ -std=c++14 -Wall -L/home/gnii/devel/catch2 -lcatch test_istream.cpp && a.out
///
#define  CATCH_CONFIG_MAIN
#include "../../../catch2/catch.hpp"
#define  __GPU__
#define  __INLINE__
#define  CUEF_USE_STRING   1

#include "../src/string.h"
#include "../src/istream.h"

using namespace cuef;

TEST_CASE("istream class")
{
    const char *aa[] = {
        "abc def",
        "  abc  def ",
        " \tabc def",
        "abcdefghi jklmnopqr"
    };
    int nn = sizeof(aa)/sizeof(char*);
    const char *rst[nn][2] = {
        { "abc", "def" },
        { "abc", "def" },
        { "abc", "def" },
        { "abcdefghi", "jklmnopqr" }
    };
    SECTION("is(int),str(char*,sz),str(string&),gcount(),tellg()") {
        istream v1(16);
        REQUIRE(v1.gcount()==16);
        REQUIRE(v1.tellg()==0);
        istream v2(13);
        REQUIRE(v2.gcount()==16);
        REQUIRE(v2.tellg()==0);
        istream v3;
        v3.str((char*)aa[3]);
        REQUIRE(v3.gcount()==ALIGN4(strlen(aa[3])));
        REQUIRE(v3.tellg()==0);
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
    SECTION(">>(char*)") {
        char buf[20];
        istream v1;
        for (int i=0; i<nn; i++) {
            v1.str(aa[i]);
            v1 >> buf;
            REQUIRE(strcmp(buf, rst[i][0])==0);
            v1 >> buf;
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
    SECTION(">>(string&)") {
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
