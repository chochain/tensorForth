# Makefile for TensorBoard FlatBuffers Demo
# Requires: g++ with C++17 support

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Iinclude

SRCS     = src/main.cpp
TARGET   = tb_demo

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^
	@echo ""
	@echo "Build successful: ./$(TARGET)"
	@echo "Run with:         ./$(TARGET) [logdir]"

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
