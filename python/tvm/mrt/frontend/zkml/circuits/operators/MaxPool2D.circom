pragma circom 2.0.3;

include "../circomlib-matrix/matElemSum.circom";
include "../util.circom";

template Max1D (n) {
    signal input in[n];
    signal output out;
    component gts[n];        // store comparators
    component switchers[n+1];  // switcher for comparing maxs

    signal maxs[n+1];

    maxs[0] <== in[0];
    for(var i = 0; i < n; i++) {
        gts[i] = GreaterThan_Full();
        switchers[i+1] = Switcher();

        gts[i].b <== maxs[i];
        gts[i].a <== in[i];

        switchers[i+1].sel <== gts[i].out;
        switchers[i+1].L <== maxs[i];
        switchers[i+1].R <== in[i];

        maxs[i+1] <== switchers[i+1].outL;
    }

    out <== maxs[n];
}


template MaxPool2D (C, H, W, P, S) {
    signal input in[C][H][W];
    signal output out[C][(H-P)\S+1][(W-P)\S+1];

    component max1D[C][(H-P)\S+1][(W-P)\S+1];

    for (var k = 0; k < C; k++) {
        for (var i = 0; i < (H-P)\S+1; i++) {
            for (var j = 0; j < (W-P)\S+1; j++) {
                max1D[k][i][j] = Max1D(P*P);
                for (var x = 0; x < P; x++) {
                    for (var y = 0; y < P; y++) {
                        max1D[k][i][j].in[x*P+y] <== in[k][i*S+x][j*S+y];
                    }
                }

                out[k][i][j] <== max1D[k][i][j].out;
            }
        }
    }

}
