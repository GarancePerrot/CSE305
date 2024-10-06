# compiler
CXX = g++

# source file
SOURCE = main.cpp

# output 
OUTPUT = test

# default target executed when calling make
all:
	$(CXX) $(SOURCE) -o $(OUTPUT)

# clean up
clean:
	rm -f $(OUTPUT)
