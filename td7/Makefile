CXX = g++
CFLAGS = -pthread -std=c++11 -Wall

SOURCES = gradinglib/gradinglib.cpp grading/grading.cpp main.cpp 
OBJECTS = gradinglib.o grading.o main.o 
STUDENTS_SOURCES = CoarseSetList.cpp SetList.cpp td7.cpp

grader: $(OBJECTS)
	$(CXX) $(CFLAGS) -o grader $(OBJECTS) 

gradinglib.o: gradinglib/gradinglib.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CFLAGS) -o gradinglib.o gradinglib/gradinglib.cpp

grading.o: grading/grading.cpp gradinglib/gradinglib.hpp CoarseSetList.cpp SetList.cpp td7.cpp
	$(CXX) -c $(CFLAGS) -o grading.o grading/grading.cpp -I.

main.o: main.cpp grading/grading.hpp
	$(CXX) -c $(CFLAGS) -o main.o main.cpp

set_benchmarker: CoarseSetList.cpp SetList.cpp benchmarking_sets.cpp 
	$(CXX) $(CFLAGS) -o set_benchmarker benchmarking_sets.cpp

clean:
	rm -f *.o
	rm -f grader
	rm -f set_benchmarker
