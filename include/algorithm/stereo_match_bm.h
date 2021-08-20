#include "algorithm/stereo_match.h"

namespace stereolab
{
namespace algorithm
{

struct BlockMatchParam
{

    BlockMatchParam::BlockMatchParam()
        : min_disp(0), max_disp(64), block_size(3), cost_type(CostType::kSAD)
    {
    }

    int32_t min_disp;
    int32_t max_disp;

    int32_t block_size;

    CostType cost_type;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    BlockMatchParam,
    min_disp,
    max_disp,
    block_size,
    cost_type)

class StereoMatchBM : public StereoMatch
{
    constexpr static float kInvalidCost = 1.0;

public:
    virtual bool configure(const nlohmann::json &config) override;

    virtual bool compute(const common::StereoData &data, cv::Mat &disp) override;

private:
    BlockMatchParam param_;
};

}  // namespace algorithm
}  // namespace stereolab