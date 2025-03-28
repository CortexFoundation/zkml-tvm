pragma circom 2.1.0;

template Element1DAdd(iShape) {
    signal input A[iShape];
    signal input B[iShape];
    signal output out[iShape];

    for (var i=0; i < iShape; i++) {
        out[i] <== A[i] + B[i];
    }
}

template Element2DAdd(i1, i2) {
    signal input A[i1][i2];
    signal input B[i1][i2];
    signal output out[i1][i2];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            out[i][j] <== A[i][j] + B[i][j];
        }
    }
}

template Element3DAdd(i1, i2, i3) {
    signal input A[i1][i2][i3];
    signal input B[i1][i2][i3];
    signal output out[i1][i2][i3];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                out[i][j][k] <== A[i][j][k] + B[i][j][k];
            }
        }
    }
}

template Element1DSub(iShape) {
    signal input A[iShape];
    signal input B[iShape];
    signal output out[iShape];

    for (var i=0; i < iShape; i++) {
        out[i] <== A[i] - B[i];
    }
}

template Element2DSub(i1, i2) {
    signal input A[i1][i2];
    signal input B[i1][i2];
    signal output out[i1][i2];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            out[i][j] <== A[i][j] - B[i][j];
        }
    }
}

template Element3DSub(i1, i2, i3) {
    signal input A[i1][i2][i3];
    signal input B[i1][i2][i3];
    signal output out[i1][i2][i3];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                out[i][j][k] <== A[i][j][k] - B[i][j][k];
            }
        }
    }
}

template Element1DMul(iShape) {
    signal input A[iShape];
    signal input B[iShape];
    signal output out[iShape];

    for (var i=0; i < iShape; i++) {
        out[i] <== A[i] * B[i];
    }

}

template Negative3D (i1, i2, i3) {
    signal input in[i1][i2][i3];
    signal output out[i1][i2][i3];
    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                out[i][j][k] <== -1 *in[i][j][k];
            }
        }
    }
}

