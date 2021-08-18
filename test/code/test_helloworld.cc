#include <gtest/gtest.h>

#include "common/info_log.h"

TEST(PRINT_TEST, PRINT_HELLOWORLD)
{
    SL_PRINTE("helloworld!!!\n");
    SL_PRINTW("helloworld!!!\n");
    SL_PRINTD("helloworld!!!\n");
    SL_PRINTI("helloworld!!!\n");
    
    ASSERT_TRUE(true);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}