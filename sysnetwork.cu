// 
// System network function definitions
// 
// 2022, Jonathan Tainer
// 

#include "network.h"
#include <cuda.h>

Network::Network() {
	layer = NULL;
	numOfLayers = 0;
	numOfInputs = 0;
}

Network::Network(const Network& source) {
	numOfLayers = source.numOfLayers;
	numOfInputs = source.numOfInputs;
	layer = new Layer[numOfLayers];
	for (int i = 0; i < numOfLayers; i++) {
		layer[i] = source.layer[i];
	}
}

Network::Network(const DevNetwork& source) {
	numOfLayers = source.numOfLayers;
	numOfInputs = source.numOfInputs;
	layer = new Layer[numOfLayers];
	for (int i = 0; i < numOfLayers; i++) {
		layer[i] = source.layer[i];
	}
}

Network::Network(const int inputs, const int outputs, const int layers, const int nodes) {
	numOfLayers = layers;
	numOfInputs = inputs;
	layer = new Layer[numOfLayers];
	layer[0] = Layer(nodes, inputs);
	for (int i = 1; i < layers - 1; i++) {
		layer[i] = Layer(nodes, nodes);
	}
	layer[layers - 1] = Layer(outputs, nodes);
}

Network& Network::operator=(const Network& source) {
	if (this == &source) {
		return *this;
	}
	
	// Reallocate memory only if needed
	if (numOfLayers != source.numOfLayers) {
		numOfLayers = source.numOfLayers;
		delete[] layer;
		layer = new Layer[numOfLayers];
	}
	
	for (int i = 0; i < numOfLayers; i++) {
		layer[i] = source.layer[i];
	}
	
	numOfInputs = source.numOfInputs;
	
	return *this;
}

Network& Network::operator=(const DevNetwork& source) {
	
	// Reallocate memory only if needed
	if (numOfLayers != source.numOfLayers) {
		numOfLayers = source.numOfLayers;
		delete[] layer;
		layer = new Layer[numOfLayers];
	}
	
	
	
	for (int i = 0; i < numOfLayers; i++) {
		layer[i] = source.layer[i];
	}
	
	numOfInputs = source.numOfInputs;
	
	return *this;
}

void Network::randomize(const float min, const float max) {
	for (int i = 0; i < numOfLayers; i++) {
		layer[i].randomize(min, max);
	}
}

Network::~Network() {
	delete[] layer;
}
