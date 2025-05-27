pragma circom 2.1.0;

// 2D
template Broadcast2DAxis0Sub (i1, i2) {
    signal input A[i1][i2];
    signal input B[i1][1];
    signal output out[i1][i2];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            out[i][j] <== A[i][j] - B[i][0];
        }
    }
}

template Broadcast2DAxis0Add (i1, i2) {
    signal input A[i1][i2];
    signal input B[i1][1];
    signal output out[i1][i2];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            out[i][j] <== A[i][j] + B[i][0];
        }
    }
}

template Broadcast2DAxis1Sub (i1, i2) {
    signal input A[i1][i2];
    signal input B[1][i1][1];
    signal output out[i1][i2];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            out[i][j] <== A[i][j] - B[0][i][0];
        }
    }
}

template Broadcast2DAxis1Add (i1, i2) {
    signal input A[i1][i2];
    signal input B[1][i1][1];
    signal output out[i1][i2];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            out[i][j] <== A[i][j] + B[0][i][0];
        }
    }
}

// 3D
template Broadcast3DAxis0Sub (i1, i2, i3) {
    signal input A[i1][i2][i3];
    signal input B[i1][1][1];
    signal output out[i1][i2][i3];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                out[i][j][k] <== A[i][j][k] - B[i][0][0];
            }
        }
    }
}

template Broadcast3DAxis0Add (i1, i2, i3) {
    signal input A[i1][i2][i3];
    signal input B[i1][1][1];
    signal output out[i1][i2][i3];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                out[i][j][k] <== A[i][j][k] + B[i][0][0];
            }
        }
    }
}

template Broadcast3DAxis1Sub (i1, i2, i3) {
    signal input A[i1][i2][i3];
    signal input B[1][i1][1][1];
    signal output out[i1][i2][i3];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                out[i][j][k] <== A[i][j][k] - B[0][i][0][0];
            }
        }
    }
}

template Broadcast3DAxis1Add (i1, i2, i3) {
    signal input A[i1][i2][i3];
    signal input B[1][i1][1][1];
    signal output out[i1][i2][i3];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                out[i][j][k] <== A[i][j][k] + B[0][i][0][0];
            }
        }
    }
}

template Broadcast3DAxis1Add_0_2 (i1, i2, i3) {
    signal input A[i1][i2][i3];
    signal input B[1][i1][1][i3];
    signal output out[i1][i2][i3];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                out[i][j][k] <== A[i][j][k] + B[0][i][0][k];
            }
        }
    }
}


template Broadcast4DAxis0Add (i1, i2, i3, i4) {
    signal input A[i1][i2][i3][i4];
    signal input B[i1][1][1][1];
    signal output out[i1][i2][i3][i4];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                for (var g = 0; g < i4; g++) {
                    out[i][j][k][g] <== A[i][j][k][g] + B[i][0][0][0];
                }
            }
        }
    }
}

template Broadcast4DAxis1Add (i1, i2, i3, i4) {
    signal input A[i1][i2][i3][i4];
    signal input B[1][i1][1][1][1];
    signal output out[i1][i2][i3][i4];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                for (var g = 0; g < i4; g++) {
                    out[i][j][k][g] <== A[i][j][k][g] + B[0][i][0][0][0];
                }
            }
        }
    }
}

template Broadcast4DAxis1Add_First (i1, i2, i3, i4) {
    signal input A[i1][i2][i3][i4];
    signal input B[i1][1][1][1][1];
    signal output out[i1][i2][i3][i4];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                for (var g = 0; g < i4; g++) {
                    out[i][j][k][g] <== A[i][j][k][g] + B[i][0][0][0][0];
                }
            }
        }
    }
}

