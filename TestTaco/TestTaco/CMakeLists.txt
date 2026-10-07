add_executable(TestTaco
    "TestTaco.cpp"
    "DataGenerator.cpp"
    "TensorBuilder.cpp"
    "MemoryInfo.cpp"
)

target_include_directories(TestTaco PRIVATE
    "/mnt/f/Kruk/Разработка/taco/include"
)

target_link_directories(TestTaco PRIVATE
    "/mnt/f/Kruk/Разработка/taco/out/build/WSL-GCC-Debug/lib"
)

find_package(OpenMP REQUIRED)

target_link_libraries(TestTaco PRIVATE
    taco
    OpenMP::OpenMP_CXX
)

set_property(TARGET TestTaco PROPERTY CXX_STANDARD 20)