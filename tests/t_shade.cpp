#include <stdio.h>

static const char *lk = " `.-:;!+*ixekO#@      ";     // 16 shades
static const int   sz = 16;

void shade(int n) {
    for (int i=0; i<20; i++) {
        printf("%c", lk[n]);
    }
}

int main(int argc, char **argv) {
    for (int y=0; y<5; y++) {
        for (int j=0; j<16; j++) {
            for (int x=0; x<5; x++) {
                int s = y * 4 + x;
                shade(s);
            }
            printf("\n");
        }
    }
    return 0;
}
    
