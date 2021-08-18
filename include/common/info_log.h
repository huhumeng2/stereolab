#pragma once

#include <cstring>

#include "common/config.h"

namespace stereolab
{
namespace common
{

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#define FILE_NAME(x) strrchr(x,'\\')?strrchr(x,'\\')+1:x
#else
#define FILE_NAME(x) strrchr(x,'/')?strrchr(x,'/')+1:x
#endif
#define COLOR_NONE         "\033[0m"
#define FONT_RED           "\033[0;31m"
#define FONT_MAG           "\033[0;35m"
#define BG_COLOR_RED       "\033[41m"
#define BG_RED_FONT_YELLOW "\033[41;33m"

#define SL_PRINT_RAW(Level, _fmt, ...)   \
    {                                    \
        if constexpr(Level <= LOG_LEVEL)           \
        {                                \
            printf(_fmt, ##__VA_ARGS__); \
        }                                \
    }

#define SL_PRINTE(Fmt, ...)                          \
    {                                                \
        printf(BG_COLOR_RED BG_RED_FONT_YELLOW);     \
        printf("[E](%s:%d): ", FILE_NAME(__FILE__), __LINE__);  \
        SL_PRINT_RAW(LOG_ERROR, Fmt, ##__VA_ARGS__); \
        printf(COLOR_NONE);                          \
    }

#define SL_PRINTW(Fmt, ...)                         \
    {                                               \
        printf(FONT_RED);                           \
        printf("[W](%s:%d): ",  FILE_NAME(__FILE__), __LINE__); \
        SL_PRINT_RAW(LOG_WARN, Fmt, ##__VA_ARGS__); \
        printf(COLOR_NONE);                         \
    }

#define SL_PRINTD(Fmt, ...)                          \
    {                                                \
        printf(FONT_MAG);                            \
        printf("[D](%s:%d): ",  FILE_NAME(__FILE__), __LINE__);  \
        SL_PRINT_RAW(LOG_DEBUG, Fmt, ##__VA_ARGS__); \
        printf(COLOR_NONE);                          \
    }

#define SL_PRINTI(Fmt, ...)                         \
    {                                               \
        printf("[I](%s:%d): ",  FILE_NAME(__FILE__), __LINE__); \
        SL_PRINT_RAW(LOG_INFO, Fmt, ##__VA_ARGS__); \
    }

}  // namespace common
}  // namespace stereolab