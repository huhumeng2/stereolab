#include <opencv2/core/core.hpp>

namespace stereolab
{
namespace algorithm
{

struct DisparityPlane
{
    DisparityPlane(float nx, float ny, float nz, float z)
        : nx(nx), ny(ny), nz(nz), z(z) {}

    float disparity(float x, float y) const { return a * x + b * y + c; }
    
    void update_coeff(float x, float y, float d)
    {
        float inv_z = 1.0f / nz;
        a = -nx * inv_z;
        b = -ny * inv_z;
        c = (nx * x + ny * y) * inv_z + d;
    }

    float a, b, c;
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