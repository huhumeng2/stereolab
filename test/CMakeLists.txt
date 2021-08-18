message(STATUS "Build Test")

find_package(GTest REQUIRED)
add_executable(test_helloworld ${PROJECT_SOURCE_DIR}/test/code/test_helloworld.cc)
target_include_directories(
    test_helloworld
    PRIVATE
    ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(
    test_helloworld
    PRIVATE
    ${GTEST_LIBRARIES}
)