# Compiler settings
CXX      = g++
NVCC     = nvcc
CXXFLAGS = -O3 -std=c++17
NVCCFLAGS = -O3 -std=c++17

# Output
TARGET   = md_sim
BUILD_DIR = build

# Source files
SRCS = src/main.cu src/md_kernel.cu src/io.cpp src/utils.cpp src/vtk_writer.cpp
OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cu=.o)

# Default target
.PHONY: all
all: $(BUILD_DIR) $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR):
    mkdir -p $(BUILD_DIR)

# Build rule for the main executable
$(BUILD_DIR)/$(TARGET): $(SRCS)
    $(NVCC) $(NVCCFLAGS) -o $@ $(SRCS)

# Clean rule
.PHONY: clean
clean:
    rm -rf $(BUILD_DIR) output/*.vtk

# Run example (customize as needed)
.PHONY: run
run: all
    ./$(BUILD_DIR)/$(TARGET) input/particles_test1.txt 0.001 1000 1.0 1.0
