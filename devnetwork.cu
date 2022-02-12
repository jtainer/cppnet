// 
// Device network function definitions
// 
// 2022, Jonathan Tainer
// 

#include "network.h"
#include "matmul.h"
#include <cuda.h>

DevNetwork::DevNetwork() {
	layer = NULL;
	devInput = NULL;
	numOfLayers = 0;
}

DevNetwork::DevNetwork(const DevNetwork& source) {
	numOfLayers = source.numOfLayers;
	layer = new DevLayer[numOfLayers];
	for (int i = 0; i < numOfLayers; i++) {
		layer[i] = source.layer[i];
	}
	
	numOfInputs = source.numOfInputs;
	cudaMalloc((void**)&devInput, sizeof(float) * numOfInputs);
}

DevNetwork::DevNetwork(const Network& source) {
	numOfLayers = source.numOfLayers;
	layer = new DevLayer[numOfLayers];
	for (int i = 0; i < numOfLayers; i++) {
		layer[i] = source.layer[i];
	}
	
	numOfInputs = source.numOfInputs;
	cudaMalloc((void**)&devInput, sizeof(float) * numOfInputs);
}

DevNetwork& DevNetwork::operator=(const DevNetwork& source) {
	if (this == &source) {
		return *this;
	}
	
	// Reallocate memory only if needed
	if (numOfLayers != source.numOfLayers) {
		numOfLayers = source.numOfLayers;
		delete[] layer;
		layer = new DevLayer[numOfLayers];
	}
	
	if (numOfInputs != source.numOfInputs) {
		numOfInputs = source.numOfInputs;
		cudaFree(devInput);
		cudaMalloc((void**)&devInput, sizeof(float) * numOfInputs);
	}
	
	for (int i = 0; i < numOfLayers; i++) {
		layer[i] = source.layer[i];
	}
	
	return *this;
}

DevNetwork& DevNetwork::operator=(const Network& source) {
	
	// Reallocate memory only if needed
	if (numOfLayers != source.numOfLayers) {
		numOfLayers = source.numOfLayers;
		delete[] layer;
		layer = new DevLayer[numOfLayers];
	}
	
	if (numOfInputs != source.numOfInputs) {
		numOfInputs = source.numOfInputs;
		cudaFree(devInput);
		cudaMalloc((void**)&devInput, sizeof(float) * numOfInputs);
	}
	
	for (int i = 0; i < numOfLayers; i++) {
		layer[i] = source.layer[i];
	}
	
	return *this;
}

void DevNetwork::forwardPass(float* inputVector) {
	
	// Copy input vector to device
	cudaMemcpy(devInput, inputVector, sizeof(float) * numOfInputs, cudaMemcpyHostToDevice);
	
	float* tmpInputVector = devInput;
	
	for (int i = 0; i < numOfLayers; i++) {
		matmul<<<(layer[i].numOfNodes / 256) + 1, 256>>>(tmpInputVector, layer[i]);
		tmpInputVector = layer[i].outputVector;
	}
}

void DevNetwork::forwardPass(float* inputVector, float* outputVector) {
	
	this->forwardPass(inputVector);
	
	// Copy output vector to system memory
	cudaMemcpy(outputVector, layer[numOfLayers - 1].outputVector, sizeof(float) * layer[numOfLayers - 1].numOfNodes, cudaMemcpyDeviceToHost);
}

DevNetwork::~DevNetwork() {
	delete[] layer;
	cudaFree(devInput);
}
