pragma circom 2.1.0;

template Transpose2D (i1, i2) {
    signal input in[i1][i2];
    // 2,1
    signal output out[i2][i1];

    for (var x1 = 0; x1 < i1; x1++) {
        for (var x2 = 0; x2 < i2; x2++) {
            // 2,1
            out[x2][x1] <== in[x1][x2];
        }
    }
}

template Transpose3D_231 (i1, i2, i3) {
    signal input in[i1][i2][i3];
    // 2,3,1
    signal output out[i2][i3][i1];

    for (var x1 = 0; x1 < i1; x1++) {
        for (var x2 = 0; x2 < i2; x2++) {
            for (var x3 = 0; x3 < i3; x3++) {
                // 2,3,1
                out[x2][x3][x1] <== in[x1][x2][x3];
            }
        }
    }
}

template Transpose3D_312 (i1, i2, i3) {
    signal input in[i1][i2][i3];
    // 3,1,2
    signal output out[i3][i1][i2];

    for (var x1 = 0; x1 < i1; x1++) {
        for (var x2 = 0; x2 < i2; x2++) {
            for (var x3 = 0; x3 < i3; x3++) {
                // 3,1,2
                out[x3][x1][x2] <== in[x1][x2][x3];
            }
        }
    }
}

template Transpose4D_2134 (i1, i2, i3, i4) {
    signal input in[i1][i2][i3][i4];
    // 2,1,3,4
    signal output out[i2][i1][i3][i4];

    for (var x1 = 0; x1 < i1; x1++) {
        for (var x2 = 0; x2 < i2; x2++) {
            for (var x3 = 0; x3 < i3; x3++) {
                for (var x4 = 0; x4 < i4; x4++) {
                    // 2,1,3,4
                    out[x2][x1][x3][x4] <== in[x1][x2][x3][x4];
                }
            }
        }
    }

}
