#include "../src/ten4_types.h"

void dump(DU A[3][3], DU I[3][3], int sz) {
    for (int y=0; y<sz; y++) {
        for (int x=0; x<sz; x++) printf(" %f", A[y][x]);
        printf(" | ");
        for (int x=0; x<sz; x++) printf(" %f", I[y][x]);
        printf("\n");
    }                
}
int  find_max(DU A[3][3], int z, int sz) {
    int u = z;
    /* finding maximum xth column element in last (sz-x) rows */
    for (int y = z + 1; y < sz; y++) {
        if (A[y][z] > A[u][z]) u = y;
    }
    if (fabs(A[u][z]) < DU_EPS) {
        printf("sigular !!!\n");
        return -1;
    }
    else printf("A[%d][%d]=%f\n", u, z, A[u][z]);
    
    return u;
}
void swap_row(DU A[3][3], int z, int u, int sz) {
    for (int k=0; k<sz; k++) {      // swap entire row
        float ta = A[z][k];
        A[z][k] = A[u][k];
        A[u][k] = ta;
    }
}
void h_gj0(DU A[3][3], DU I[3][3], int z, int sz) {
    for (int y = 0; y < sz; y++) {
        float r0 = A[y][z], r1 = r0 / A[z][z];
        for (int k = 0; k < sz; k++) {
            if (y==z) {                  // divide the working row
                I[y][k] /= r0;
                A[y][k] /= r0;
            }
            else {
                I[y][k] -= r1 * I[z][k]; // subtract all other rows
                A[y][k] -= r1 * A[z][k];
            }
        }
    }
}
void h_gj(DU A[3][3], DU I[3][3], int z, int sz) {       // TODO: block-wise
    float r0 = A[z][z];
    for (int k = 0; k < sz; k++) {
        I[z][k] /= r0;
        A[z][k] /= r0;
    }
    for (int y = 0; y < sz; y++) {
        float r1 = A[y][z];
        for (int k = 0; y!=z && k < sz; k++) {
            I[y][k] -= r1 * I[z][k]; // subtract all other rows
            A[y][k] -= r1 * A[z][k];
        }
    }
}
void h_tri_u(DU A[3][3], int z, int sz) {
    float r0 = A[z][z];
//    for (int k = 0; k < sz; k++) A[z][k] /= r0;
    // upper
    for (int y = z + 1; y < sz; y++) {
        float r1 = A[y][z] / r0;
        for (int k = 0; k < sz; k++) {
            A[y][k] -= r1 * A[z][k];
        }
    }
}
void h_tri_l(DU A[3][3], int z, int sz) {
    float r0 = A[z][z];
//    for (int k = 0; k < sz; k++) A[z][k] /= r0;
    // lower
    for (int x = z + 1; x < sz; x++) {
        float r1 = A[z][x] / r0;
        for (int k = 0; k < sz; k++) {
            A[k][x] -= r1 * A[k][z];
        }
    }
}
void h_tri_lu(DU A[3][3], DU I[3][3], int z, int sz) {
    float ra = A[z][z], ri = I[z][z];
    for (int k = 0; k < sz; k++) {
        A[z][k] /= ra;
        I[z][k] /= ri;
    }
    // upper
    for (int y = z + 1; y < sz; y++) {
        float r1 = A[y][z];
        for (int k = 0; k < sz; k++) {
            A[y][k] -= r1 * A[z][k];
        }
    }
    // lower
    for (int x = z + 1; x < sz; x++) {
        float r1 = I[z][x];
        for (int k = 0; k < sz; k++) {
            I[k][x] -= r1 * I[k][z];
        }
    }
}
int lu(DU A[3][3], DU I[3][3], int sz) {
	for (int z = 0; z < sz; z++) {
        printf("==== %d\n", z);
        dump(A, I, sz);
        
        int u = find_max(A, z, sz), v = find_max(I, z, sz);
        if (u < 0 || v < 0) return 0;
        
		if (u != z) swap_row(A, z, u, sz);
        if (v != z) swap_row(I, z, v, sz);

        dump(A, I, sz);

        h_tri_u(A, z, sz);
        h_tri_l(I, z, sz);
//          h_tri_lu(A, I, z, sz);
//        dim3 block(16,16);
//        dim3 grid((sz + block.x - 1) / block.x, (sz + block.y - 1) / block.y);
//        k_gj<<<grid, block>>>((DU*)A, (DU*)I, z, sz);
//        cudaDeviceSynchronize();
	}
    return 1;
}
void h_gj2(DU A[3][3], DU I[3][3], int z, int sz) {
    int y = 0, k = 0;
    /*
    while (k < sz && y < sz) {
        float r0 = A[z + y * sz], r1 = r0 / zz;
        int   yk = k + y * sz;
        int   zk = k + z * sz;
        if (y==z) {              // divide the working row
            I[yk] /= r0;
            A[yk] /= r0;
        }
        else {
            I[yk] -= r1 * I[yk]; // subtract all other rows
            A[yk] -= r1 * A[yk];
        }
        if (++k >= sz) { y++; k=0; }
    }
    */
}    
__KERN__ void k_gj(DU *A, DU *I, int z, int sz) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float zz = A[z + z * sz];   // A[z][z]
    if (k < sz && y < sz) {
        float r0 = A[z + y * sz], r1 = r0 / zz;
        int   yk = k + y * sz;
        int   zk = k + z * sz;
        if (y==z) {
            I[yk] /= r0;
            A[yk] /= r0;
        }
        else {
            I[yk] -= r1 * I[zk];
            A[yk] -= r1 * A[zk];
        }
    }
}
int gj(DU A[3][3], DU I[3][3], int sz) {
	for (int z = 0; z < sz; z++) {
        printf("==== %d\n", z);
        dump(A, I, sz);
        
        int u = find_max(A, z, sz);
        if (u < 0) return 0;
        
		if (u != z) { 		// swapping row which has maximum xth column element
            swap_row(A, z, u, sz);
            swap_row(I, z, u, sz);
        }
        dump(A, I, sz);

        h_gj(A, I, z, sz);
//        dim3 block(16,16);
//        dim3 grid((sz + block.x - 1) / block.x, (sz + block.y - 1) / block.y);
//        k_gj<<<grid, block>>>((DU*)A, (DU*)I, z, sz);
//        cudaDeviceSynchronize();
	}
    return 1;
}
int main(int argc, char **argv) {
    DU A[3][3] = {{1,1,2},{2,1,1},{1,2,1}};
    DU U[3][3] = {{1,1,2},{2,1,1},{1,2,1}};
    DU I[3][3] = {{1,0,0},{0,1,0},{0,0,1}};

    gj(A, I, 3);
//    h_lu(A, U, 3);
    printf("final=====\n");
    dump(A, I, 3);

    return 0;
}
