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
int h_gj(DU A[3][3], DU I[3][3], int z, int sz) {       // TODO: block-wise
    float r0 = A[z][z];
    if (fabs(r0) < DU_EPS) return -1; // skip the row
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
    return 0;
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
//
// A    = PLU
// A^-1 = (PLU)^-1
// A^-1 = (U)^-1(PL)^-1
// A^-1 = (U)^-1(L)^-1(P)^-1
// A^-1 = (U)^-1(L)^-1(P)t
//
void h_tri(DU A[3][3], int z, int sz) {
    float ra = A[z][z];
    if (fabs(ra) < DU_EPS) return;     // if 0 skip the row
    for (int y = z + 1; y < sz; y++) {
        float r1 = -A[y][z] / ra;
        for (int k = z; k < sz; k++) {
            A[y][k] += r1 * A[z][k];
        }
        A[y][z] = r1;                  // L^-1 store in U to save space
    }
}
int plu(DU A[3][3], DU P[3], int sz) {
    DU I[3][3] = {{1,0,0},{0,1,0},{0,0,1}}; // dummy
    printf("PLU = A\n");
	for (int z = 0; z < sz; z++) {
        printf("==== %d\n", z);
        dump(A, I, P, sz);
        int u = find_max(A, z, sz);    // pivot to reduce rounding error
        if (u < 0) return -1;
		if (u != z) { 	               // swapping row which has maximum xth column element
            swap_row(A, z, u, z, sz);
            DU t = P[z]; P[z] = P[u]; P[u] = t;
        }
        dump(A, I, P, sz);
        h_tri(A, z, sz);
	}
    return 0;
}
///
/// LU inversion
///
int h_lu_inv(DU A[3][3], DU I[3][3], int z, int sz) {
    float r0 = A[z][z]; A[z][z]  = 1.0;
    if (fabs(r0) < DU_EPS) return -1;             // singular
    
    for (int k = z; k < sz; k++) I[z][k] /= r0;   // current z row
    for (int y = 0; y < z; y++) {                 // factorize rows above
        float r1 = A[y][z]; A[y][z] = 0.0;
        for (int k = z; k < sz; k++) {
            I[y][k] -= I[z][k] * r1;
        }
    }
    return 0;
}
int lu_inv(DU A[3][3], DU I[3][3], DU P[3], int sz) {
    DU X[3] = { 0, 0, 0 };
    /// A => L^-1, I => U^-1
    for (int z = sz-1; z >= 0; z--) {
        printf("LU inv==== %d\n", z);
        dump(A, I, X, sz);
        
        if (h_lu_inv(A, I, z, sz) < 0) {
            printf("lu_inv sigular!!!\n");
            return -1;
        }
    }
    /// A^-1 = (U^-1)(L^-1)(PI), i.e. use P to swap_rows
    return 0;
}
int main(int argc, char **argv) {
    DU A[3][3] = {{1,1,2},{2,1,1},{1,2,1}};
//    DU A[3][3] = {{2,2,5},{1,1,1},{4,6,8}};
//    DU A[3][3] = {{6,18,-12},{2,4,-2},{3,17,10}};
    DU I[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    DU P[3]    = { 0, 1, 2 };               // compressed sparse matrix

//    gj(A, I, 3);
    plu(A, P, 3);
    lu_inv(A, I, P, 3);
    printf("final=====\n");
    dump(A, I, P, 3);

    return 0;
}
