pragma circom 2.0.3;

template BiasAdd1 (C) {
    signal input in[C];
    signal input bias[C];
    signal output out[C];

    for (var i = 0; i < C; i++) {
        out[i] <== in[i] + bias[i];
    }
}

template BiasAdd2 (C, H) {
    signal input in[C][H];
    signal input bias[C];
    signal output out[C][H];

    for (var i = 0; i < C; i++) {
        for (var j = 0; j < H; j++) {
            out[i][j] <== in[i][j] + bias[i];
        }
    }
}

template BiasAdd3 (C, H, W) {
    signal input in[C][H][W];
    signal input bias[C];
    signal output out[C][H][W];

    for (var i = 0; i < C; i++) {
        for (var j = 0; j < H; j++) {
            for (var k = 0; k < W; k++) {
                out[i][j][k] <== in[i][j][k] + bias[i];
            }
        }
    }
}
