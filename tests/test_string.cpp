///
/// Unit Test - cueForth vector
/// 
///> g++ -std=c++14 -Wall -L/home/gnii/devel/catch2 -lcatch test_string.cpp && a.out
///
#define  CATCH_CONFIG_MAIN
#include "../../../catch2/catch.hpp"

#define __GPU__
#include "../src/string.h"

using namespace cuef;

TEST_CASE("string class")
{
    char buf[80] = {0};
    auto dump = [&buf](string& s) {
        int i;
        for (i=0; i<s.size(); i++) buf[i] = s[i];
        buf[i] = '\0';
        return buf;
    };
    const char *aa[] = {
        "abc",
        "defghi",
        "jkl",
        "mnopqr"
    };
    int nn = sizeof(aa)/sizeof(int);
    const char *rst[nn] = {
        "abc",
        "abcdefghi",
        "abcdefghijkl",
        "abcdefghijklmnopqr"
    };
    SECTION("constructor(),resize") {
        string v1;
        REQUIRE(v1.size()==0);
        REQUIRE(v1._sz==STRING_BUF_SIZE);
        v1.resize(STRING_BUF_SIZE*2);
        REQUIRE(v1.size()==0);
        REQUIRE(v1._sz==STRING_BUF_SIZE*2);
    }
    SECTION("constructor(str, n)") {
        string v1(rst[1]);
        REQUIRE(v1.size()==(int)strlen(rst[1]));
        REQUIRE(v1._sz==STRING_BUF_SIZE);
        REQUIRE(strcmp(dump(v1), rst[1])==0);
        string v2(rst[3]);
        REQUIRE(v2.size()==(int)strlen(rst[3]));
        REQUIRE(v2._sz==STRING_BUF_SIZE+4);
        REQUIRE(strcmp(dump(v2), rst[3])==0);
    }
    SECTION("str(), c_str(), ==(string), ==(char*)") {
        string v1(rst[1]);
        string& v2  = v1.str();
        char    *s2 = v1.c_str();
        REQUIRE(strcmp(v2._v, rst[1])==0);
        REQUIRE(strcmp(s2, rst[1])==0);
        REQUIRE(v1==v2);
        REQUIRE(v1==s2);
    }
    SECTION("substr(i)") {
        string v1(rst[1]);
        for (int i=0; i<6; i++) {
            string &v2 = v1.substr(i);
            REQUIRE(strcmp(dump(v2), &rst[1][i])==0);
        }
    }
    SECTION("<<(const char*)") {
        string v1;
        for (int i=0; i<4; i++) {
            v1 << aa[i];
            REQUIRE(v1.size()==(int)strlen(rst[i]));
            REQUIRE(strcmp(dump(v1), rst[i])==0);
        }
    }
    SECTION("<<(string)") {
        string v1;
        for (int i=0; i<4; i++) {
            string s(aa[i]);
            v1 << s;
            REQUIRE(v1.size()==(int)strlen(rst[i]));
            REQUIRE(strcmp(dump(v1), rst[i])==0);
        }
    }
    int ii[] = { 1111, -222, 3333, -444, 5555 };
    const char *rsi[5] = {
        "1111",
        "1111-222",
        "1111-2223333",
        "1111-2223333-444",
        "1111-2223333-4445555"
    };
    SECTION("<<(int)") {
        string v1;
        for (int i=0; i<5; i++) {
            v1 << ii[i];
            REQUIRE(v1.size()==(int)strlen(rsi[i]));
            REQUIRE(strcmp(dump(v1), rsi[i])==0);
        }
    }
    float ff[] = { 1.111, -2.2, 3.4444446 };
    const char *rsf[5] = {
        "1.111000",
        "1.111000-2.200000",
        "1.111000-2.2000003.444445"
    };
    SECTION("<<(float)") {
        string v1;
        for (int i=0; i<3; i++) {
            v1 << ff[i];
            REQUIRE(strcmp(dump(v1), rsf[i])==0);
        }
    }
    SECTION("to_i") {
    }
    SECTION("to_f") {
    }
}
