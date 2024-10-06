NVCC = /usr/local/cuda/bin/nvcc

demo:
	$(NVCC) add.cu -o demo -arch=sm_60 -std=c++11 -I/usr/local/cuda/include

sum:
	$(NVCC) sum.cu -o sum -arch=sm_60 -std=c++11 -I/usr/local/cuda/include

variance:
	$(NVCC) variance.cu -o variance -arch=sm_60 -std=c++11 -I/usr/local/cuda/include

sum_distances:
	$(NVCC) sum_distances.cu -o sum_distances -arch=sm_60 -std=c++11 -I/usr/local/cuda/include

clean:
	rm -f demo
	rm -f sum_distances
	rm -f sum
	rm -f variance
