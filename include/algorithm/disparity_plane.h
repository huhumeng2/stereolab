#include <opencv2/core/core.hpp>

namespace stereolab
{
namespace algorithm
{

struct DisparityPlane
{
    DisparityPlane(float nx, float ny, float nz, float z)
        : nx(nx), ny(ny), nz(nz), z(z) {}

    float nx, ny, nz;
    float z;
};

}  // namespace algorithm
}  // namespace stereolab

namespace cv
{
template <>
class DataType<stereolab::algorithm::DisparityPlane>
{
public:
    typedef float value_type;
    typedef value_type work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;

    enum
    {
        generic_type = 0,
        depth = CV_32F,
        channels = sizeof(stereolab::algorithm::DisparityPlane) / sizeof(float),
        fmt = ( int )'f',
        type = CV_MAKETYPE(depth, channels)
    };
};
}  // namespace cv