///
/// Unit Test - cueForth vector
/// 
///> g++ -std=c++14 -Wall -L/home/gnii/devel/catch2 -lcatch test_vector.cpp && a.out
///
#include <sstream>
#define  CATCH_CONFIG_MAIN
#include "../../../catch2/catch.hpp"

#define __GPU__
#define __INLINE__

#include "../src/vector.h"

using namespace cuef;

TEST_CASE("vector class")
{
    auto dump = [](vector<int>& a) {
        std::stringstream ss("");
        for (int i=0; i<a.size(); i++) ss << a[i] << '|';
        return ss.str();
    };
    int aa[] = { 111, 222, 333, 444, 555, 666 };
    int nn = sizeof(aa)/sizeof(int);
    std::string rst[nn+1] = {
        "",
        "111|",
        "111|222|",
        "111|222|333|",
        "111|222|333|444|",
        "111|222|333|444|555|",
        "111|222|333|444|555|666|"
    };
    SECTION("constructor/destructor/resize") {
        vector<int> v1;
        REQUIRE(v1._n==0);
        REQUIRE(v1._sz==0);
        REQUIRE(v1.size()==0);
        v1.resize(4);
        REQUIRE(v1._n==0);
        REQUIRE(v1._sz==4);
        REQUIRE(v1.size()==0);
    }
    SECTION("push/pop") {
        vector<int> v1;
        v1.push(aa[0]);
        REQUIRE(v1._n==1);
        REQUIRE(v1._sz==4);
        REQUIRE(v1.size()==1);
        REQUIRE(dump(v1)==rst[1]);
        int x = v1.pop();
        REQUIRE(v1._n==0);
        REQUIRE(v1._sz==4);
        REQUIRE(v1.size()==0);
        REQUIRE(x==aa[0]);
        REQUIRE(dump(v1)==rst[0]);
    }
    SECTION("multiple pushs") {
        vector<int> v1;
        for (int i=0, j=1; i<nn; i++, j++) {
            v1.push(aa[i]);
            REQUIRE(v1._n==j);
            REQUIRE(v1._sz==4*((i/4)+1));
            REQUIRE(v1.size()==j);
            REQUIRE(dump(v1)==rst[j]);
        }
    }
    SECTION("concate pushs/pops") {
        vector<int> v1;
        v1.push(aa[0]).push(aa[1]).push(aa[2]).push(aa[3]).push(aa[4]).push(aa[5]);
        REQUIRE(v1._n==6);
        REQUIRE(v1._sz==8);
        REQUIRE(v1.size()==6);
        REQUIRE(dump(v1)==rst[6]);
        int x;
        for (int i=nn-1; i>=0; i--) {
            x = v1.pop();
            REQUIRE(x==aa[i]);
            REQUIRE(v1._n==i);
            REQUIRE(v1._sz==8);
            REQUIRE(v1.size()==i);
            REQUIRE(dump(v1)==rst[i]);
        }
    }
    SECTION("merge, +=") {
        vector<int> v1;
        v1.merge(aa, nn);
        REQUIRE(v1._sz==8);
        REQUIRE(v1.size()==nn);
        REQUIRE(dump(v1)==rst[nn]);
        vector<int> v2;
        v2.merge(v1);
        REQUIRE(v2._sz==8);
        REQUIRE(v2.size()==nn);
        REQUIRE(dump(v2)==rst[nn]);
        vector<int> v3;
        v3 += v1;
        REQUIRE(v3._sz==8);
        REQUIRE(v3.size()==nn);
        REQUIRE(dump(v3)==rst[nn]);
    }
    SECTION("=") {
        vector<int> v1;
        v1.merge(aa, nn);
        REQUIRE(v1._sz==8);
        REQUIRE(v1.size()==nn);
        REQUIRE(dump(v1)==rst[nn]);
        vector<int>& v2 = v1;          // alias
        REQUIRE(v2._sz==8);
        REQUIRE(v2.size()==nn);
        REQUIRE(dump(v2)==rst[nn]);
        vector<int> v3 = v1;           // call constructor
        REQUIRE(v3._sz==8);
        REQUIRE(v3.size()==nn);
        REQUIRE(dump(v3)==rst[nn]);
        vector<int> v4;
        v4 = v1;                       // call operator=
        REQUIRE(v4._sz==8);
        REQUIRE(v4.size()==nn);
        REQUIRE(dump(v4)==rst[nn]);
    }
    SECTION("dec_i") {
        vector<int> v1;
        std::string r("111|222|333|444|555|665|");
        v1.merge(aa, nn);
        int x = v1.dec_i();
        REQUIRE(x==665);
        REQUIRE(v1._sz==8);
        REQUIRE(v1.size()==nn);
        REQUIRE(dump(v1)==r);
    }
    SECTION("clear") {
        vector<int> v1;
        std::string r("111|222|333|");
        v1.merge(aa, nn);
        v1.clear(3);
        REQUIRE(v1._sz==8);
        REQUIRE(v1.size()==3);
        REQUIRE(dump(v1)==rst[3]);
        v1.clear();
        REQUIRE(v1._sz==8);
        REQUIRE(v1.size()==0);
        REQUIRE(dump(v1)==rst[0]);
    }
}
