message(STATUS "Build Test")

set(TEST_DATA_PATH ${PROJECT_SOURCE_DIR}/test/data/)

configure_file(
    ${PROJECT_SOURCE_DIR}/test/test_config.h.in
    ${PROJECT_SOURCE_DIR}/test/test_config.h
)

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

add_executable(test_util ${PROJECT_SOURCE_DIR}/test/code/test_util.cc)
target_include_directories(
    test_util
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(
    test_util
    PRIVATE
    sl_common
    ${GTEST_LIBRARIES}
)

add_test(
    NAME test_helloworld 
    COMMAND test_helloworld
)

add_test(
    NAME test_util 
    COMMAND test_util
)