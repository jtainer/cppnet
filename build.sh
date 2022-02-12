nvcc -c syslayer.cu sysnetwork.cu devlayer.cu devnetwork.cu matmul.cu
g++ -c example.cpp
g++ -L/usr/local/cuda/lib64 example.o syslayer.o sysnetwork.o devlayer.o devnetwork.o matmul.o -lcudart
