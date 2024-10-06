CXX = g++
CFLAGS = -pthread -std=c++11 -Wall
NVCC = /usr/local/cuda/bin/nvcc

SOURCES = gradinglib/gradinglib.cpp grading/grading.cpp main.cpp 
OBJECTS = gradinglib.o grading.o main.o 

grader: $(OBJECTS)
	$(CXX) $(CFLAGS) -o grader $(OBJECTS) 

gradinglib.o: gradinglib/gradinglib.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CFLAGS) -o gradinglib.o gradinglib/gradinglib.cpp

grading.o: grading/grading.cpp gradinglib/gradinglib.hpp td5.cpp
	$(CXX) -c $(CFLAGS) -o grading.o grading/grading.cpp -I.

main.o: main.cpp grading/grading.hpp
	$(CXX) -c $(CFLAGS) -o main.o main.cpp

pde:
	$(NVCC) difference_scheme.cu -o pde -arch=sm_60 -std=c++11 -I/usr/local/cuda/include


clean:
	rm -f *.o
	rm -f grader
	rm -f pde
