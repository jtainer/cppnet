// 
// Structs to handle layers and networks in system memory and device memory
//
// 2022, Jonathan Tainer
//

#include <stdlib.h>
#include <string.h>

#ifndef LAYER
#define LAYER

struct Layer;
struct DevLayer;

struct Layer {

		Layer();
		Layer(const Layer& source);
		Layer(const DevLayer& source);
		Layer(const int nodes, const int inputs);
		Layer& operator=(const Layer& source);
		Layer& operator=(const DevLayer& source);
		void randomize(float min, float max);
		~Layer();

		float* weightMatrix;
		float* outputVector;
		float* delta;
		int numOfNodes;
		int weightsPerNode;
};

struct DevLayer {

		DevLayer();
		DevLayer(const DevLayer& source);
		DevLayer(const Layer& source);
		DevLayer(const int nodes, const int inputs);
		DevLayer& operator=(const DevLayer& source);
		DevLayer& operator=(const Layer& source);
		~DevLayer();

		float* weightMatrix;
		float* outputVector;
		float* delta;
		int numOfNodes;
		int weightsPerNode;
};

#endif

#ifndef NETWORK
#define NETWORK

struct Network;
struct DevNetwork;

struct Network {

		Network();
		Network(const Network& source);
		Network(const DevNetwork& source);
		Network(const int inputs, const int outputs, const int layers, const int nodes);
		Network& operator=(const Network& source);
		Network& operator=(const DevNetwork& source);
		void randomize(const float min, const float max);
		~Network();

		Layer* layer;
		int numOfLayers;
		int numOfInputs;
};

struct DevNetwork {

		DevNetwork();
		DevNetwork(const DevNetwork& source);
		DevNetwork(const Network& source);
		DevNetwork& operator=(const DevNetwork& source);
		DevNetwork& operator=(const Network& source);
		void forwardPass(float* inputVector);
		void forwardPass(float* inputVector, float* outputVector);
		
		~DevNetwork();

		DevLayer* layer;
		float* devInput;
		int numOfLayers;
		int numOfInputs;
};

#endif
