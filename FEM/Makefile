CXX ?= g++
CPPFLAGS := -Iinclude
CXXFLAGS ?= -Wall -Wextra -Wpedantic -std=c++17 -O3

TARGET := generate_pinn_data.x
SOURCES := src/generate_pinn_data.cpp src/fem.cpp src/mesh.cpp
OBJECTS := $(patsubst src/%.cpp,build/%.o,$(SOURCES))
DEPS := $(OBJECTS:.o=.d)

.DEFAULT_GOAL := all

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@

build/%.o: src/%.cpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

build:
	mkdir -p $@

clean:
	rm -rf build $(TARGET)

-include $(DEPS)

.PHONY: all clean
