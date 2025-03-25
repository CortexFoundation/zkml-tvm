// from 0xZKML/zk-mnist

pragma circom 2.0.3;

include "../circomlib/comparators.circom";
include "../circomlib/switcher.circom";
include "../util.circom";

template ArgMax (n) {
    signal input in[n];
    signal output out;
    component gts[n];        // store comparators
    component switchers[n+1];  // switcher for comparing maxs
    component aswitchers[n+1]; // switcher for arg max

    signal maxs[n+1];
    signal amaxs[n+1];

    maxs[0] <== in[0];
    amaxs[0] <== 0;
    for(var i = 0; i < n; i++) {
        gts[i] = GreaterThan_Full();
        switchers[i+1] = Switcher();
        aswitchers[i+1] = Switcher();

        gts[i].b <== maxs[i];
        gts[i].a <== in[i];

        switchers[i+1].sel <== gts[i].out;
        switchers[i+1].L <== maxs[i];
        switchers[i+1].R <== in[i];

        aswitchers[i+1].sel <== gts[i].out;
        aswitchers[i+1].L <== amaxs[i];
        aswitchers[i+1].R <== i;
        amaxs[i+1] <== aswitchers[i+1].outL;
        maxs[i+1] <== switchers[i+1].outL;
    }

    out <== amaxs[n];
}

template Greater2D (i1, i2) {
    signal input in1[i1][i2];
    signal input in2;
    signal output out[i1][i2];

    component gts[i1][i2];
    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            gts[i][j] = GreaterThan_Full();
            gts[i][j].a <== in1[i][j];
            gts[i][j].b <== in2;
	    out[i][j] <== gts[i][j].out;
        }
    }
}

template Where2D(i1, i2) {
    signal input sel[i1][i2];
    signal input in1[i1][i2];
    signal input in2[i1][i2];
    signal output out[i1][i2];

    component switchers[i1][i2];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            switchers[i][j] = Switcher();
            switchers[i][j].sel <== sel[i][j];
            switchers[i][j].L <== in1[i][j];
            switchers[i][j].R <== in2[i][j];
	    out[i][j] <== switchers[i][j].outL;
        }
    }
}

