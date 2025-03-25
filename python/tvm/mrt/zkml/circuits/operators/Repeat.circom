pragma circom 2.1.0;

template Repeat2D_0A (i1, i2, times) {
    signal input in[i1][i2];
    signal output out[i1*times][i2];
    for (var i = 0; i < times*i1; i++) {
        for (var j = 0; j < i2; j++) {
            out[i][j] <== in[i%i1][j];
        }
    }
}

template Repeat2D_1A (i1, i2, times) {
    signal input in[i1][i2];
    signal output out[i1][i2*times];
    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < times*i2; j++) {
            out[i][j] <== in[i][j%i2];
        }
    }
}

template Repeat3D_0A (i1, i2, i3, times) {
    signal input in[i1][i2][i3];
    signal output out[i1*times][i2][i3];
    for (var i = 0; i < times*i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                out[i][j][k] <== in[i%i1][j][k];
            }
        }
    }
}

template Repeat3D_1A (i1, i2, i3, times) {
    signal input in[i1][i2][i3];
    signal output out[i1][i2*times][i3];
    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2*times; j++) {
            for (var k = 0; k < i3; k++) {
                out[i][j][k] <== in[i][j%i2][k];
            }
        }
    }
}

template Repeat3D_2A (i1, i2, i3, times) {
    signal input in[i1][i2][i3];
    signal output out[i1][i2][i3*times];
    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3*times; k++) {
                out[i][j][k] <== in[i][j][k%i3];
            }
        }
    }
}

template ExpandDims3D_3A (i1, i2, i3, new_axis) {
    signal input in[i1][i2][i3];
    signal output out[i1][i2][i3][new_axis];
    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                for (var g = 0; g < new_axis; g++) {
                    out[i][j][k][g] <== in[i][j][k];
                }
            }
        }
    }
}

template Tile3D (i1, i2, i3, times) {
    signal input in[i1][i2][i3];
    signal output out[times][i1][i2][i3];
    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                for (var t = 0; t < times; t++) {
                  out[t][i][j][k] <== in[i][j][k];
                }
            }
        }
    }
}

template Tile4D (i1, i2, i3, i4, times) {
    signal input in[i1][i2][i3][i4];
    signal output out[times][i1][i2][i3][i4];
    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
              for (var g = 0; g < i4; g++) {
                for (var t = 0; t < times; t++) {
                  out[t][i][j][k][g] <== in[i][j][k][g];
                }
              }
            }
        }
    }
}

