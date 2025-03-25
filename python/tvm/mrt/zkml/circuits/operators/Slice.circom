pragma circom 2.1.0;

template StrideSlice2D (shp1, shp2, b1, b2, e1, e2, stride) {
    signal input in[shp1][shp2];
    signal output out[(e1-b1)\stride][(e2-b2)\stride];
    for (var i = b1; i < e1; i+=stride) {
        for (var j = b2; j < e2; j+=stride) {
	    out[(i-b1)\stride][(j-b2)\stride] <== in[i][j];
        }
    }
}

template StrideSlice3D (shp1, shp2, shp3, b1, b2, b3, e1, e2, e3, stride) {
    signal input in[shp1][shp2][shp3];
    signal output out[(e1-b1)\stride][(e2-b2)\stride][(e3-b3)\stride];
    for (var i = b1; i < e1; i+=stride) {
        for (var j = b2; j < e2; j+=stride) {
          for (var k = b3; k < e3; k+=stride) {
           out[(i-b1)\stride][(j-b2)\stride][(k-b3)\stride] <== in[i][j][k];
          }
        }
    }
}

template SliceLike3D_2_3 (i1, i2, i3, i4, i5, i6) {
    signal input in1[i1][i2][i3];
    signal input in2[i4][i5][i6];
    signal output out[i1][i5][i6];
    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i5 && j<i2; j+=1) {
          for (var k = 0; k < i6 && k<i3; k+=1) {
            out[i][j][k] <== in1[i][j][k];
          }
        }
    }
}

