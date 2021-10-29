add_executable(demo Demo.cc)

target_link_libraries(demo tiangongClassification ${OpenCV_LIBS})

target_include_directories(demo PRIVATE ${OpenCV_INCLUDE_DIRS})