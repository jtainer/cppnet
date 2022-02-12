// 
// System layer member function definitions
//
// 2022, Jonathan Tainer
//

#include "network.h"
#include <cuda.h>

Layer::Layer() {
	weightMatrix = NULL;
	outputVector = NULL;
	delta = NULL;
	numOfNodes = 0;
	weightsPerNode = 0;
}

Layer::Layer(const Layer& source) {
	numOfNodes = source.numOfNodes;
	weightsPerNode = source.weightsPerNode;
	
	weightMatrix = (float*)malloc(numOfNodes * weightsPerNode * sizeof(float));
	outputVector = (float*)malloc(numOfNodes * sizeof(float));
	delta = (float*)malloc(numOfNodes * sizeof(float));

	memcpy(weightMatrix, source.weightMatrix, numOfNodes * weightsPerNode * sizeof(float));
	memcpy(outputVector, source.outputVector, numOfNodes * sizeof(float));
	memcpy(delta, source.delta, numOfNodes * sizeof(float));
}

Layer::Layer(const DevLayer& source) {
	numOfNodes = source.numOfNodes;
	weightsPerNode = source.weightsPerNode;

	weightMatrix = (float*)malloc(numOfNodes * weightsPerNode * sizeof(float));
	outputVector = (float*)malloc(numOfNodes * sizeof(float));
	delta = (float*)malloc(numOfNodes * sizeof(float));

	cudaMemcpy(weightMatrix, source.weightMatrix, numOfNodes * weightsPerNode * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(outputVector, source.outputVector, numOfNodes * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(delta, source.delta, numOfNodes * sizeof(float), cudaMemcpyDeviceToHost);
}

Layer::Layer(const int nodes, const int inputs) {
	numOfNodes = nodes;
	weightsPerNode = inputs;

	weightMatrix = (float*)malloc(nodes * inputs * sizeof(float));
	outputVector = (float*)malloc(nodes * sizeof(float));
	delta = (float*)malloc(nodes * sizeof(float));
}

Layer& Layer::operator=(const Layer& source) {
	if (this == &source) {
		return *this;
	}
	
	// Reallocate memory only if needed
	if (numOfNodes != source.numOfNodes || weightsPerNode != source.weightsPerNode) {
		free(weightMatrix);
		free(outputVector);
		free(delta);

		numOfNodes = source.numOfNodes;
		weightsPerNode = source.weightsPerNode;

		weightMatrix = (float*)malloc(numOfNodes * weightsPerNode * sizeof(float));
		outputVector = (float*)malloc(numOfNodes * sizeof(float));
		delta = (float*)malloc(numOfNodes * sizeof(float));
	}

	memcpy(weightMatrix, source.weightMatrix, numOfNodes * weightsPerNode * sizeof(float));
	memcpy(outputVector, source.outputVector, numOfNodes * sizeof(float));
	memcpy(delta, source.delta, numOfNodes * sizeof(float));

	return *this;
}

Layer& Layer::operator=(const DevLayer& source) {
	
	// Reallocate memory only if needed
	if (numOfNodes != source.numOfNodes || weightsPerNode != source.weightsPerNode) {
		free(weightMatrix);
		free(outputVector);
		free(delta);

		numOfNodes = source.numOfNodes;
		weightsPerNode = source.weightsPerNode;

		weightMatrix = (float*)malloc(numOfNodes * weightsPerNode * sizeof(float));
		outputVector = (float*)malloc(numOfNodes * sizeof(float));
		delta = (float*)malloc(numOfNodes * sizeof(float));
	}
	
	cudaMemcpy(weightMatrix, source.weightMatrix, numOfNodes * weightsPerNode * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(outputVector, source.outputVector, numOfNodes * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(delta, source.delta, numOfNodes * sizeof(float), cudaMemcpyDeviceToHost);

	return *this;
}

void Layer::randomize(const float min, const float max) {
	
	float multiplier = (max - min) / RAND_MAX;
	int matSize = numOfNodes * weightsPerNode;

	for (int i = 0; i < matSize; i++) {
		weightMatrix[i] = (rand() * multiplier) - min;
	}
}


Layer::~Layer() {
	free(weightMatrix);
	free(outputVector);
	free(delta);
}
