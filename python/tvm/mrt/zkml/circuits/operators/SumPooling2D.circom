pragma circom 2.0.3;

include "../circomlib-matrix/matElemSum.circom";
include "../util.circom";

// SumPooling2D layer, basically AveragePooling2D layer with a constant scaling, more optimized for circom, strides=poolSize like Keras default
template SumPooling2D (nRows, nCols, nChannels, poolSize) {
    signal input in[nRows][nCols][nChannels];
    signal output out[nRows\poolSize][nCols\poolSize][nChannels];

    component elemSum[nRows\poolSize][nCols\poolSize][nChannels];

    for (var i=0; i<nRows\poolSize; i++) {
        for (var j=0; j<nCols\poolSize; j++) {
            for (var k=0; k<nChannels; k++) {
                elemSum[i][j][k] = matElemSum(poolSize,poolSize);
                for (var x=0; x<poolSize; x++) {
                    for (var y=0; y<poolSize; y++) {
                        elemSum[i][j][k].a[x][y] <== in[i*poolSize+x][j*poolSize+y][k];
                    }
                }
                out[i][j][k] <== elemSum[i][j][k].out;
            }
        }
    }
}

// CHW, sum of H and W, keep H,W as 1,1 (keep_dims = True, or None)
template Sum_CHW (C, H, W) {
    signal input in[C][H][W];
    signal output out[C][1][1];

    component elemSum[C];

    for (var i=0; i<C; i++) {
        elemSum[i] = matElemSum(H, W);
	for (var x=0; x<H; x++) {
	    for (var y=0; y<W; y++) {
		elemSum[i].a[x][y] <== in[i][x][y];
	    }
	}
	out[i][0][0] <== elemSum[i].out;
    }
}

// CHW, sum of H and W, (keep_dims=False)
template Sum_CHW_0 (C, H, W) {
    signal input in[C][H][W];
    signal output out[C];

    component elemSum[C];

    for (var i=0; i<C; i++) {
        elemSum[i] = matElemSum(H, W);
	for (var x=0; x<H; x++) {
	    for (var y=0; y<W; y++) {
		elemSum[i].a[x][y] <== in[i][x][y];
	    }
	}
	out[i] <== elemSum[i].out;
    }
}
