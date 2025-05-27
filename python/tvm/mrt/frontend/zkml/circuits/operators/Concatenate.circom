pragma circom 2.1.0;

template Concatenate1D0A (i1_0, i1_1) {
    signal input in0[i1_0];
    signal input in1[i1_1];
    signal output out[i1_0 + i1_1];

    for (var i = 0; i < i1_0; i++) {
	out[i] <== in0[i];
    }
    for (var i = 0; i < i1_1; i++) {
	out[i+i1_0] <== in1[i];
    }
}

template Concatenate2D0A (i1_0, i1_1, i2) {
    signal input in0[i1_0][i2];
    signal input in1[i1_1][i2];
    signal output out[i1_0 + i1_1][i2];

    for (var j = 0; j < i2; j++) {
        for (var i = 0; i < i1_0; i++) {
            out[i][j] <== in0[i][j];
        }
        for (var i = 0; i < i1_1; i++) {
            out[i+i1_0][j] <== in1[i][j];
        }
    }
}

template Concatenate2D1A (i1, i2_0, i2_1) {
    signal input in0[i1][i2_0];
    signal input in1[i1][i2_1];
    signal output out[i1][i2_0 + i2_1];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2_0; j++) {
            out[i][j] <== in0[i][j];
        }
        for (var j = 0; j < i2_1; j++) {
            out[i][j+i2_0] <== in1[i][j];
        }
    }
}

template Concatenate3D0A (i1_0, i1_1, i2, i3) {
    signal input in0[i1_0][i2][i3];
    signal input in1[i1_1][i2][i3];
    signal output out[i1_0 + i1_1][i2][i3];

    for (var j = 0; j < i2; j++) {
        for (var k = 0; k < i3; k++) {
            for (var i = 0; i < i1_0; i++) {
                out[i][j][k] <== in0[i][j][k];
            }
            for (var i = 0; i < i1_1; i++) {
                out[i+i1_0][j][k] <== in1[i][j][k];
            }
        }
    }
}

template Concatenate3D2A (i1, i2, i3_0, i3_1) {
    signal input in0[i1][i2][i3_0];
    signal input in1[i1][i2][i3_1];
    signal output out[i1][i2][i3_0 + i3_1];

    for (var i = 0; i < i1; i++) {
      for (var j = 0; j < i2; j++) {
        for (var k = 0; k < i3_0; k++) {
            out[i][j][k] <== in0[i][j][k];
        }
        for (var k = 0; k < i3_1; k++) {
            out[i][j][k+i3_0] <== in1[i][j][k];
        }
      }
    }
}

template Concatenate4D3A (i1, i2, i3, i4_0, i4_1) {
    signal input in0[i1][i2][i3][i4_0];
    signal input in1[i1][i2][i3][i4_1];
    signal output out[i1][i2][i3][i4_0 + i4_1];

    for (var i = 0; i < i1; i++) {
      for (var j = 0; j < i2; j++) {
        for (var k = 0; k < i3; k++) {
          for (var g = 0; g < i4_0; g++) {
              out[i][j][k][g] <== in0[i][j][k][g];
          }
          for (var g = 0; g < i4_1; g++) {
              out[i][j][k][g+i4_0] <== in1[i][j][k][g];
          }
        }
      }
    }
}

