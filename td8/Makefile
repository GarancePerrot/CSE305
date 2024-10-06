CXX = g++
CFLAGS = -pthread -std=c++14 -Wall

SOURCES = gradinglib/gradinglib.cpp grading/grading.cpp main.cpp 
OBJECTS = gradinglib.o grading.o main.o 
STUDENTS_SOURCES = td8.cpp

grader: $(OBJECTS)
	$(CXX) $(CFLAGS) -o grader $(OBJECTS) -latomic

gradinglib.o: gradinglib/gradinglib.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CFLAGS) -o gradinglib.o gradinglib/gradinglib.cpp

grading.o: grading/grading.cpp gradinglib/gradinglib.hpp td8.cpp
	$(CXX) -c $(CFLAGS) -o grading.o grading/grading.cpp -I.

main.o: main.cpp grading/grading.hpp
	$(CXX) -c $(CFLAGS) -o main.o main.cpp

clean:
	rm -f *.o
	rm -f grader
