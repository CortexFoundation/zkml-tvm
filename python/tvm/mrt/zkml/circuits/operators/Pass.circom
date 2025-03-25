pragma circom 2.1.0;

include "../circomlib/comparators.circom";
include "../circomlib/switcher.circom";

template Pass1D (i1) {
    signal input in[i1];
    signal output out[i1];
    for (var i = 0; i < i1; i++) {
	out[i] <== in[i];
    }
}

template Pass2D (i1, i2) {
    signal input in[i1][i2];
    signal output out[i1][i2];
    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
	    out[i][j] <== in[i][j];
        }
    }
}

template Pass3D (i1, i2, i3) {
    signal input in[i1][i2][i3];
    signal output out[i1][i2][i3];
    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                out[i][j][k] <== in[i][j][k];
            }
        }
    }
}

template Pass4D (i1, i2, i3, i4) {
    signal input in[i1][i2][i3][i4];
    signal output out[i1][i2][i3][i4];
    for (var x1 = 0; x1 < i1; x1++) {
        for (var x2 = 0; x2 < i2; x2++) {
            for (var x3 = 0; x3 < i3; x3++) {
                for (var x4 = 0; x4 < i4; x4++) {
                    out[x1][x2][x3][x4] <== in[x1][x2][x3][x4];
                }
            }
        }
    }
}

template Pass5D (i1, i2, i3, i4, i5) {
    signal input in[i1][i2][i3][i4][i5];
    signal output out[i1][i2][i3][i4][i5];
    for (var x1 = 0; x1 < i1; x1++) {
        for (var x2 = 0; x2 < i2; x2++) {
            for (var x3 = 0; x3 < i3; x3++) {
                for (var x4 = 0; x4 < i4; x4++) {
                    for (var x5 = 0; x5 < i5; x5++) {
                        out[x1][x2][x3][x4][x5] <== in[x1][x2][x3][x4][x5];
                    }
                }
            }
        }
    }
}

template Split3D (i1, i2, i3) {
    // deprecated, should pass
    signal input in[i1][i2][i3];
    signal output out1[i1\2][i2][i3];
    signal output out2[i1\2][i2][i3];
}

// dimension=2, split 0-axis
template TupleGetItem2D0A (i1, i2, parts, index) {
    signal input in[i1][i2];
    signal output out[i1\parts][i2];
    for (var i = index * i1\parts; i < index * i1\parts + i1\parts; i++) {
        for (var j = 0; j < i2; j++) {
	    out[i-index * i1\parts][j] <== in[i][j];
        }
    }
}

// dimension=2, split 1-axis
template TupleGetItem2D1A (i1, i2, parts, index) {
    signal input in[i1][i2];
    signal output out[i1][i2\parts];
    for (var j = index * i2\parts; j < index * i2\parts + i2\parts; j++) {
        for (var i = 0; i < i1; i++) {
	    out[i][j-index * i2\parts] <== in[i][j];
        }
    }
}

// dimension=3, split 0-axis
template TupleGetItem3D0A (i1, i2, i3, parts, index) {
    signal input in[i1][i2][i3];
    signal output out[i1\parts][i2][i3];
    for (var i = index * i1\parts; i < index * i1\parts + i1\parts; i++) {
      for (var j = 0; j < i2; j++) {
        for (var k = 0; k < i3; k++) {
          out[i-index * i1\parts][j][k] <== in[i][j][k];
        }
      }
    }
}

// dimension=3, split 1-axis
template TupleGetItem3D1A (i1, i2, i3, parts, index) {
    signal input in[i1][i2][i3];
    signal output out[i1][i2\parts][i3];
    for (var i = 0; i < i1; i++) {
      for (var j = index * i2\parts; j < index * i2\parts + i2\parts; j++) {
        for (var k = 0; k < i3; k++) {
          out[i][j-index * i2\parts][k] <== in[i][j][k];
        }
      }
    }
}

// dimension=3, split 2-axis
template TupleGetItem3D2A (i1, i2, i3, parts, index) {
    signal input in[i1][i2][i3];
    signal output out[i1][i2][i3\parts];
    for (var i = 0; i < i1; i++) {
      for (var j = 0; j < i2; j++) {
        for (var k = index * i3\parts; k < index * i3\parts + i3\parts; k++) {
          out[i][j][k-index * i3\parts] <== in[i][j][k];
        }
      }
    }
}

template Tuple3Item(count) {
    signal input in_class[count][1];
    signal input in_score[count][1];
    signal input in_anchor[count][4];
    signal output out[count][6];
    for (var i = 0; i < count; i++) {
      out[i][0] <== in_class[i][0];
      out[i][1] <== in_score[i][0];
      out[i][2] <== in_anchor[i][0];
      out[i][3] <== in_anchor[i][1];
      out[i][4] <== in_anchor[i][2];
      out[i][5] <== in_anchor[i][3];
    }
}

template AdvIndex_select_funcutil (i1) {
    signal input in[i1];
    signal input index;
    signal output out;

    component isequal[i1];
    component switcher[i1];
    var tempout[i1+1];
    tempout[0] === 0;

    for (var i = 0; i < i1; i++) {
        isequal[i] = IsEqual();
        isequal[i].in[0] <== i;
        isequal[i].in[1] <== index;

        switcher[i] = Switcher();
        switcher[i].sel <== isequal[i].out;
        switcher[i].L <== in[i];
        switcher[i].R <== tempout[i];
        tempout[i+1] === switcher[i].outL;
    }
    out <== tempout[i1];
}

template AdvIndex2D (i1, j1, j2) {
    signal input inp[i1];
    signal input index[j1][j2];
    signal output out[j1][j2];

    component advIndex_select_funcutil_0[j1][j2];

    for (var i = 0; i < j1; i++) {
        for (var j = 0; j < j2; j++) {
            advIndex_select_funcutil_0[i][j] = AdvIndex_select_funcutil(i1);
            for (var x = 0; x < i1; x++) {
                advIndex_select_funcutil_0[i][j].in[x] <== inp[x];
                advIndex_select_funcutil_0[i][j].index <== index[i][j];
            }
	    out[i][j] <== advIndex_select_funcutil_0[i][j].out;
        }
    }
}

template AdvIndex3D (i1, j1, j2, j3) {
    signal input inp[i1];
    signal input index[j1][j2][j3];
    signal output out[j1][j2][j3];

    component advIndex_select_funcutil_0[j1][j2][j3];

    for (var i = 0; i < j1; i++) {
        for (var j = 0; j < j2; j++) {
            for (var k = 0; k < j3; k++) {
              advIndex_select_funcutil_0[i][j][k] = AdvIndex_select_funcutil(i1);
              for (var x = 0; x < i1; x++) {
                advIndex_select_funcutil_0[i][j][k].in[x] <== inp[x];
                advIndex_select_funcutil_0[i][j][k].index <== index[i][j][k];
              }
	      out[i][j][k] <== advIndex_select_funcutil_0[i][j][k].out;
            }
        }
    }
}
