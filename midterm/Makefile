CXX = g++
CFLAGS = -pthread -std=c++14 -Wall
NVCC = /usr/local/cuda/bin/nvcc


ex1: ex1.cpp
	$(CXX) $(CFLAGS) -o ex1 ex1.cpp

ex3: ex3.cu
	$(NVCC) ex3.cu -o ex3 -arch=sm_60 -std=c++11 -I/usr/local/cuda/include

ex4: ex4.cpp
	$(CXX) $(CFLAGS) -o ex4 ex4.cpp

clean:
	rm -f ex1
	rm -f ex3
	rm -f ex4
