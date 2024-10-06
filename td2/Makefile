CXX = g++
CFLAGS = -pthread -std=c++11 -Wall

SOURCES = gradinglib/gradinglib.cpp grading/grading.cpp main.cpp 
OBJECTS = gradinglib.o grading.o main.o 

grader: $(OBJECTS)
	# uncomment for the computer room
	# export LD_LIBRARY_PATH=/usr/local/gcc-12.1.0/lib64:$LD_LIBRARY_PATH
	$(CXX) $(CFLAGS) -o grader $(OBJECTS) 

gradinglib.o: gradinglib/gradinglib.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CFLAGS) -o gradinglib.o gradinglib/gradinglib.cpp

grading.o: grading/grading.cpp gradinglib/gradinglib.hpp td2.cpp
	$(CXX) -c $(CFLAGS) -o grading.o grading/grading.cpp -I.

main.o: main.cpp grading/grading.hpp
	$(CXX) -c $(CFLAGS) -o main.o main.cpp

clean:
	rm -f *.o
	rm -f grader
