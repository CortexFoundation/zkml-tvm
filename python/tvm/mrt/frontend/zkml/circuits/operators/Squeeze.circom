pragma circom 2.1.0;

// C,H,W = C,1,1 to C
template Squeeze_CHW (C, H, W) {
    signal input A[C][H][W];
    signal output out[C];

    for (var i = 0; i < C; i++) {
	out[i] <== A[i][0][0];
    }
}
