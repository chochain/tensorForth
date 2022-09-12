#include <iostream>    // std::cout
#include <fstream>     // std::ifstream

using namespace std;

typedef unsigned int  U32;
typedef unsigned char U8;

int show(U8 *img, int H, int W, int v) {
    static const char *map = " .:-=+*#%@";
    for (int i = 0; i < H; i++) {
        printf("\n");
        for (int j = 0; j < W; j++) {
            printf("%c", *(map + (*img++ / 26)));
        }
    }
    printf(" label=%d\n", v);
    return 0;
}

int fetch() {
    auto get_n = [](ifstream &fs) {
        U32 v = 0;
        char x;
        for (int i = 0; i < 4; i++) {
            fs.read(&x, 1);
            v <<= 8;
            v += (U32)*(U8*)&x;
        }
        return v;
    };
    ifstream icin("/u01/data/mnist/train-images-idx3-ubyte", ios::binary);
    if (!icin.is_open()) return -1;
    
    U32 X = get_n(icin);
    U32 N = get_n(icin);
    U32 H = get_n(icin);
    U32 W = get_n(icin);

    printf("MNIST image: magic=%08x,[%d][%d,%d]\n", X, N, H, W);
    U8 **img_lst = new U8*[N];
    for (int n = 0; n < N; n++) {
        img_lst[n] = new U8[H * W];
        icin.read((char*)img_lst[n], H * W);
    }
    icin.close();

    icin.open("/u01/data/mnist/train-labels-idx1-ubyte", ios::binary);
    if (!icin.is_open()) return -2;
    
    X = get_n(icin), N = get_n(icin);
    printf("MNIST label: magic=%08x,[%d]\n", X, N);
    U8 *label_lst = new U8[N];
    for (int n = 0; n < N; n++) {
        icin.read((char*)label_lst, N);
    }
    icin.close();

    for (int n = 0; n < 10; n++) {
        show(img_lst[n], H, W, (int)label_lst[n]);
    }

    return 0;
}

int main(int argc, char **argv) {
    return fetch();
}

