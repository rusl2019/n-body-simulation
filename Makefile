# Compiler
NVCC = nvcc

# C++ Compiler
CXX = g++

# Source files
CU_SOURCES = src/utils.cu src/simulation.cu
CPP_SOURCES = src/main.cpp src/sphere.cpp src/camera.cpp src/graphics.cpp

# Object files
CU_OBJS = $(CU_SOURCES:.cu=.o)
CPP_OBJS = $(CPP_SOURCES:.cpp=.o)
OBJS = $(CU_OBJS) $(CPP_OBJS)

# Executable name
TARGET = n_body_simulation

# Include paths
INCLUDES = -I./include

# Libraries
LIBS = -lGL -lGLEW -lglfw

# Compiler flags
CXXFLAGS = -std=c++11 -Wall
NVCCFLAGS = -Xcompiler "-fno-gnu-unique"

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $(TARGET) $(LIBS)

# Rule to compile .cpp files
src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Rule to compile .cu files
src/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Clean rule
clean:
	rm -f src/*.o $(TARGET)
