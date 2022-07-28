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
    /*
    for (int y = z + 1; y < sz; y++) {
        if (fabs(A[y][z]) > fabs(A[u][z])) u = y;
    }
    */
    if (fabs(A[u][z]) < DU_EPS) {
        printf("sigular !!!\n");
        return -1;
    }
    else printf("max A[%d][%d]=%f\n", u, z, A[u][z]);
    
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
void h_elim(DU A[3][3], int z, int sz) { // outer product gauss elimination
    float ra = A[z][z];
    if (fabs(ra) < DU_EPS) return;       // skip row that is 0
    for (int y = z + 1; y < sz; y++) {
        float r1 = A[y][z] / ra;
        for (int k = z; k < sz; k++) {
            A[y][k] -= r1 * A[z][k];
        }
        A[y][z] = r1;                    // L store in U to save space
    }
}
int lu(DU A[3][3], DU P[3], int sz) {    // LU with permutation
    DU I[3][3] = {{1,0,0},{0,1,0},{0,0,1}}; // dummy
    printf("PLU = A\n");
	for (int z = 0; z < sz; z++) {
        printf("==== %d\n", z);
        dump(A, I, P, sz);
        int u = find_max(A, z, sz);      // pivot to reduce rounding error
        if (u < 0) return -1;            // singular matrix
		if (u != z) { 	                 // swapping row which has maximum xth column element
            swap_row(A, z, u, z, sz);
            DU t = P[z]; P[z] = P[u]; P[u] = t;
        }
        dump(A, I, P, sz);
        h_elim(A, z, sz);
	}
    return 0;
}
///
/// LU inversion
///
float h_det(DU A[3][3], int sz) {
    float d = 1.0;
    for (int y = 0; y < sz; y++) {
        d *= A[y][y];
    }
    printf("LU det= %f\n", d);
    return d;
}
void h_forward(DU A[3][3], int z, int sz) {
    // lower triangle forward substitution
    for (int y = z + 1; y < sz; y++) {              //
        float r1 = A[y][z];
        for (int k = 0; k < z; k++) {               // columns before
            A[y][k] -= A[z][k] * r1;
        }
        A[y][z] = -r1;                              // current z column
    }
}
void h_backward(DU A[3][3], int z, int sz) {
    // upper triangle backward substitution
    float r0 = 1.0/A[z][z];
    A[z][z] = r0;                                   // diag
    for (int k = z + 1; k < sz; k++) {              // current z row
        A[z][k] *= r0;
    }
    for (int y = 0; y < z; y++) {                   // factorize rows above
        float r1 = A[y][z];
        A[y][z] = -r1 * r0;                         // current z column
        for (int k = z + 1; k < sz; k++) {          // columns after
            A[y][k] -= A[z][k] * r1;
        }
    }
}
int lu_inv(DU A[3][3], DU P[3], int sz) {
    if (fabs(h_det(A, sz)) < DU_EPS) return -1;  // singular

    /// I += L^-1
    for (int z = 0; z < sz - 1; z++) {           // forward lower triangle
        printf("forward ==== %d\n", z);
        h_forward(A, z, sz);                       
        dump(A, A, P, sz);
    }
    /// I += U^-1
    for (int z = sz-1; z >= 0; z--) {            // backward upper triangle
        printf("backward ==== %d\n", z);
        h_backward(A, z, sz);
        dump(A, A, P, sz);
    }
    /// A^-1 = (U^-1)(L^-1)(PI), i.e. use P to swap_rows
    return 0;
}
int main(int argc, char **argv) {
//    DU A[3][3] = {{1,1,2},{2,1,1},{1,2,1}};
//    DU A[3][3] = {{2,2,5},{1,1,1},{4,6,8}};
    DU A[3][3] = {{1, 2, 4},{3, 8, 14},{2, 6, 13}};
    DU I[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    DU P[3]    = { 0, 1, 2 };               // compressed sparse matrix

//    gj(A, I, 3);
    lu(A, P, 3);
    lu_inv(A, P, 3);

    return 0;
}
