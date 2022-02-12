// 
// Device layer member function definitions
// 
// 2022, Jonathan Tainer
//

#include "network.h"
#include <cuda.h>

DevLayer::DevLayer() {
	weightMatrix = NULL;
	outputVector = NULL;
	delta = NULL;
	numOfNodes = 0;
	weightsPerNode = 0;
}

DevLayer::DevLayer(const DevLayer& source) {
	numOfNodes = source.numOfNodes;
	weightsPerNode = source.weightsPerNode;

	cudaMalloc((void**)&weightMatrix, sizeof(float) * numOfNodes * weightsPerNode);
	cudaMalloc((void**)&outputVector, sizeof(float) * numOfNodes);
	cudaMalloc((void**)&delta, sizeof(float) * numOfNodes);

	cudaMemcpy(weightMatrix, source.weightMatrix, sizeof(float) * numOfNodes * weightsPerNode, cudaMemcpyDeviceToDevice);
	cudaMemcpy(outputVector, source.outputVector, sizeof(float) * numOfNodes, cudaMemcpyDeviceToDevice);
	cudaMemcpy(delta, source.delta, sizeof(float) * numOfNodes, cudaMemcpyDeviceToDevice);
}

DevLayer::DevLayer(const Layer& source) {
	numOfNodes = source.numOfNodes;
	weightsPerNode = source.weightsPerNode;

	cudaMalloc((void**)&weightMatrix, sizeof(float) * numOfNodes * weightsPerNode);
	cudaMalloc((void**)&outputVector, sizeof(float) * numOfNodes);
	cudaMalloc((void**)&delta, sizeof(float) * numOfNodes);

	cudaMemcpy(weightMatrix, source.weightMatrix, sizeof(float) * numOfNodes * weightsPerNode, cudaMemcpyHostToDevice);
	cudaMemcpy(outputVector, source.outputVector, sizeof(float) * numOfNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(delta, source.delta, sizeof(float) * numOfNodes, cudaMemcpyHostToDevice);
}

DevLayer::DevLayer(const int nodes, const int inputs) {
	numOfNodes = nodes;
	weightsPerNode = inputs;

	cudaMalloc((void**)&weightMatrix, sizeof(float) * numOfNodes * weightsPerNode);
	cudaMalloc((void**)&outputVector, sizeof(float) * numOfNodes);
	cudaMalloc((void**)&delta, sizeof(float) * numOfNodes);
}

DevLayer& DevLayer::operator=(const DevLayer& source) {
	if (this == &source) {
		return *this;
	}
	
	// Reallocate memory only if needed
	if (numOfNodes != source.numOfNodes || weightsPerNode != source.weightsPerNode) {
		cudaFree(weightMatrix);
		cudaFree(outputVector);
		cudaFree(delta);
	
		numOfNodes = source.numOfNodes;
		weightsPerNode = source.weightsPerNode;
	
		cudaMalloc((void**)&weightMatrix, sizeof(float) * numOfNodes * weightsPerNode);
		cudaMalloc((void**)&outputVector, sizeof(float) * numOfNodes);
		cudaMalloc((void**)&delta, sizeof(float) * numOfNodes);
	}
	
	cudaMemcpy(weightMatrix, source.weightMatrix, sizeof(float) * numOfNodes * weightsPerNode, cudaMemcpyDeviceToDevice);
	cudaMemcpy(outputVector, source.outputVector, sizeof(float) * numOfNodes, cudaMemcpyDeviceToDevice);
	cudaMemcpy(delta, source.delta, sizeof(float) * numOfNodes, cudaMemcpyDeviceToDevice);
	
	return *this;
}

DevLayer& DevLayer::operator=(const Layer& source) {
	
	// Reallocate memory only if needed
	if (numOfNodes != source.numOfNodes || weightsPerNode != source.weightsPerNode) {
		cudaFree(weightMatrix);
		cudaFree(outputVector);
		cudaFree(delta);
	
		numOfNodes = source.numOfNodes;
		weightsPerNode = source.weightsPerNode;
	
		cudaMalloc((void**)&weightMatrix, sizeof(float) * numOfNodes * weightsPerNode);
		cudaMalloc((void**)&outputVector, sizeof(float) * numOfNodes);
		cudaMalloc((void**)&delta, sizeof(float) * numOfNodes);
	}
	
	cudaMemcpy(weightMatrix, source.weightMatrix, sizeof(float) * numOfNodes * weightsPerNode, cudaMemcpyHostToDevice);
	cudaMemcpy(outputVector, source.outputVector, sizeof(float) * numOfNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(delta, source.delta, sizeof(float) * numOfNodes, cudaMemcpyHostToDevice);
	
	return *this;
}

DevLayer::~DevLayer() {
	cudaFree(weightMatrix);
	cudaFree(outputVector);
	cudaFree(delta);
}
