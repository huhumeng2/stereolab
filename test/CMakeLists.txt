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
    ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(
    test_util
    PRIVATE
    sl_common
    ${GTEST_LIBRARIES}
)

add_executable(test_dataloader ${PROJECT_SOURCE_DIR}/test/code/test_dataloader.cc)
target_include_directories(
    test_dataloader
    PRIVATE
    ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(
    test_dataloader
    PRIVATE
    sl_common
    ${GTEST_LIBRARIES}
)

add_executable(test_stereo_match ${PROJECT_SOURCE_DIR}/test/code/test_stereo_match.cc)
target_include_directories(
    test_stereo_match
    PRIVATE
    ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(
    test_stereo_match
    PRIVATE
    sl_algorithm
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

add_test(
    NAME test_dataloader 
    COMMAND test_dataloader
)

add_test(
    NAME test_stereo_match 
    COMMAND test_stereo_match
)