/** -*- c++ -*- */
#include "../src/ten4_types.h"

void dump(DU A[3][3], DU I[3][3], DU P[3], int sz) {
    for (int y=0; y<sz; y++) {
        for (int x=0; x<sz; x++) printf(" %f", A[y][x]);
        printf(" | ");
        for (int x=0; x<sz; x++) printf(" %f", I[y][x]);
        printf(" | %f\n", P[y]);
    }                
}
int  find_max(DU A[3][3], int z, int sz) {
    int u = z;
    /* finding maximum xth column element in last (sz-x) rows */
    for (int y = z + 1; y < sz; y++) {
        if (fabs(A[y][z]) > fabs(A[u][z])) u = y;
    }
    if (fabs(A[u][z]) < DU_EPS) {
        printf("sigular !!!\n");
        return -1;
    }
    else printf("A[%d][%d]=%f\n", u, z, A[u][z]);
    
    return u;
}
void swap_row(DU A[3][3], int z, int u, int n0, int n1) {
    for (int k=n0; k<n1; k++) {      // swap entire row
        float ta = A[z][k];
        A[z][k] = A[u][k];
        A[u][k] = ta;
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
int gj(DU A[3][3], DU I[3][3], int sz) {
    printf("GaussJordan\n");
	for (int z = 0; z < sz; z++) {
        printf("==== %d\n", z);
        dump(A, I, I[0], sz);
        
        int u = find_max(A, z, sz);
        if (u < 0) return -1;         // singular
		if (u != z) { 		          // swapping row which has maximum xth column element
            swap_row(A, z, u, z, sz);
            swap_row(I, z, u, 0, sz);
        }
        dump(A, I, I[0], sz);

        h_gj(A, I, z, sz);
	}
    return 0;
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
void h_tri_lu(DU U[3][3], DU L[3][3], int z, int sz) {
    float ra = U[z][z];
    for (int y = z + 1; y < sz; y++) {
        float r1 = U[y][z] / ra;
        for (int k = 0; k < sz; k++) {
            U[y][k] -= r1 * U[z][k];
        }
        L[y][z] = r1;                  // can store in U to save space
    }
}
//
// Ax   = b
// LUx  = b
// PLUx = Pb
//
int plu(DU U[3][3], DU L[3][3], DU P[3], int sz) {
    printf("PLU = A\n");
	for (int z = 0; z < sz; z++) {
        printf("==== %d\n", z);
        dump(U, L, P, sz);
        int u = find_max(U, z, sz);    // pivot to reduce rounding error
        if (u < 0) return -1;
		if (u != z) { 	               // swapping row which has maximum xth column element
            swap_row(U, z, u, z, sz);
            swap_row(L, z, u, 0, z);
            DU t = P[z]; P[z] = P[u]; P[u] = t;
        }
        dump(U, L, P, sz);
        h_tri_lu(U, L, z, sz);
	}
    return 0;
}
int main(int argc, char **argv) {
//    DU A[3][3] = {{1,1,2},{2,1,1},{1,2,1}};
    DU A[3][3] = {{2,2,5},{1,1,1},{4,6,8}};
    DU I[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    DU P[3]    = { 0, 1, 2 };               // compressed sparse matrix

//    gj(A, I, 3);
    plu(A, I, P, 3);
    printf("final=====\n");
    dump(A, I, P, 3);

    return 0;
}
