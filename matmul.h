// 
// Kernels for forward and backward propagation
//
// 2021, Jonathan Tainer
//


#ifndef LEARNING_RATE
#define LEARNING_RATE 0.01
#endif


#ifndef MATMUL_H
#define MATMUL_H

#include <cuda.h>
#include "network.h"


//
// Forward pass functions
//

__global__
void matmul(float* input, DevLayer layer);


//
// Back propagation functions
//

__global__
void outputLayerDelta(DevLayer outputLayer, float* target);

__global__
void hiddenLayerDelta(DevLayer hiddenLayer, Layer nextLayer);

__global__
void updateWeights(DevLayer devLayer, float* inputVector);

#endif
