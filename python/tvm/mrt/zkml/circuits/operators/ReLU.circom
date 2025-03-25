pragma circom 2.0.3;

include "../util.circom";
include "../circomlib/compconstant.circom";
include "../circomlib/switcher.circom";

// ReLU layer
template ReLU () {
    signal input in;
    signal output out;

    component isPositive = IsPositive();

    isPositive.in <== in;
    
    out <== in * isPositive.out;
}

template ReLU1D (c) {
    signal input in[c];
    signal output out[c];

    component isPositive[c];

    for (var i=0; i < c; i++) {
        isPositive[i] = IsPositive();
        isPositive[i].in <== in[i];
        out[i] <== in[i] * isPositive[i].out;
    }
}

template ReLU2D (c,h) {
    signal input in[c][h];
    signal output out[c][h];

    component isPositive[c][h];

    for (var i=0; i < c; i++) {
        for (var j=0; j < h; j++) {
            isPositive[i][j] = IsPositive();
            isPositive[i][j].in <== in[i][j];
	    out[i][j] <== in[i][j] * isPositive[i][j].out;
        }
    }
}

template ReLU3D (c,h,w) {
    signal input in[c][h][w];
    signal output out[c][h][w];

    component isPositive[c][h][w];

    for (var i=0; i < c; i++) {
        for (var j=0; j < h; j++) {
            for (var k=0; k < w; k++) {
                isPositive[i][j][k] = IsPositive();
                isPositive[i][j][k].in <== in[i][j][k];
	        out[i][j][k] <== in[i][j][k] * isPositive[i][j][k].out;
            }
        }
    }
}

template Clip1D(iShape, min, max) {
    signal input in[iShape];
    signal output out[iShape];

    component ltmin[iShape];
    component swmin[iShape];
    component ltmax[iShape];
    component swmax[iShape];
    for (var i=0; i < iShape; i++) {
        // if in[i] < min, min
        // if in[i] >= min, in[i]
        ltmin[i] = LessThan_Full();
        ltmin[i].a <== in[i];
        ltmin[i].b <== min;

        swmin[i] = Switcher();
        swmin[i].sel <== ltmin[i].out;
        swmin[i].L <== in[i];
        swmin[i].R <== min;

        // if result < max, result
        // if result >= max, max
        ltmax[i] = LessThan_Full();
        ltmax[i].a <== swmin[i].outL;
        ltmax[i].b <== max;

        swmax[i] = Switcher();
        swmax[i].sel <== ltmax[i].out;
        swmax[i].L <== max;
        swmax[i].R <== swmin[i].outL;

        out[i] <== swmax[i].outL;
    }
}

template Clip2D(iShape, H, min, max) {
    signal input in[iShape][H];
    signal output out[iShape][H];

    component ltmin[iShape][H];
    component swmin[iShape][H];
    component ltmax[iShape][H];
    component swmax[iShape][H];
    for (var i=0; i < iShape; i++) {
        for (var j=0; j < H; j++) {
            ltmin[i][j] = LessThan_Full();
            ltmin[i][j].a <== in[i][j];
            ltmin[i][j].b <== min;

            swmin[i][j] = Switcher();
            swmin[i][j].sel <== ltmin[i][j].out;
            swmin[i][j].L <== in[i][j];
            swmin[i][j].R <== min;

            ltmax[i][j] = LessThan_Full();
            ltmax[i][j].a <== swmin[i][j].outL;
            ltmax[i][j].b <== max;

            swmax[i][j] = Switcher();
            swmax[i][j].sel <== ltmax[i][j].out;
            swmax[i][j].L <== max;
            swmax[i][j].R <== swmin[i][j].outL;

            out[i][j] <== swmax[i][j].outL;
        }
    }
}

template Clip3D(iShape, H, W, min, max) {
    signal input in[iShape][H][W];
    signal output out[iShape][H][W];

    component ltmin[iShape][H][W];
    component swmin[iShape][H][W];
    component ltmax[iShape][H][W];
    component swmax[iShape][H][W];
    for (var i=0; i < iShape; i++) {
        for (var j=0; j < H; j++) {
            for (var k=0; k < W; k++) {
                ltmin[i][j][k] = LessThan_Full();
                ltmin[i][j][k].a <== in[i][j][k];
                ltmin[i][j][k].b <== min;

                swmin[i][j][k] = Switcher();
                swmin[i][j][k].sel <== ltmin[i][j][k].out;
                swmin[i][j][k].L <== in[i][j][k];
                swmin[i][j][k].R <== min;

                ltmax[i][j][k] = LessThan_Full();
                ltmax[i][j][k].a <== swmin[i][j][k].outL;
                ltmax[i][j][k].b <== max;

                swmax[i][j][k] = Switcher();
                swmax[i][j][k].sel <== ltmax[i][j][k].out;
                swmax[i][j][k].L <== max;
                swmax[i][j][k].R <== swmin[i][j][k].outL;

                out[i][j][k] <== swmax[i][j][k].outL;
            }
        }
    }
}
