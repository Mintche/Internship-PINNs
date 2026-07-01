cmake_minimum_required(VERSION 3.10)

project(PINNDataGenerator LANGUAGES CXX)

add_executable(generate_pinn_data.x
    src/generate_pinn_data.cpp
    src/fem.cpp
    src/mesh.cpp
)

target_include_directories(generate_pinn_data.x PRIVATE include)
target_compile_features(generate_pinn_data.x PRIVATE cxx_std_17)
target_compile_options(generate_pinn_data.x PRIVATE -Wall -Wextra -Wpedantic -O3)
