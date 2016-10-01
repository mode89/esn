#include <adaptive_filter_rls.h>
#include <network_nsli.h>
#include <trainer_impl.h>

namespace ESN {

    std::shared_ptr<Trainer> CreateTrainer(
        const TrainerParams & params,
        const std::shared_ptr<Network> & network)
    {
        return std::make_shared<TrainerImpl>(params, network);
    }

    TrainerImpl::TrainerImpl(
        const TrainerParams & params,
        const std::shared_ptr<Network> & network)
        : mParams(params)
        , mNetwork(std::static_pointer_cast<NetworkNSLI>(network))
        , mAdaptiveFilter(mNetwork->mParams.outputCount)
    {
        for (int i = 0; i < mNetwork->mParams.outputCount; ++i)
            mAdaptiveFilter[i] = std::make_shared<AdaptiveFilterRLS>(
                mNetwork->mParams.neuronCount,
                params.forgettingFactor,
                params.initialCovariance);
    }

    void TrainerImpl::TrainSingleOutputOnline(
        unsigned index, float value, bool force)
    {
        // Calculate output without bias and scaling
        float _value = (value - mNetwork->mOutBias(index)) /
            mNetwork->mOutScale(index);

        Eigen::VectorXf w = mNetwork->mWOut.row(index).transpose();
        if (!mNetwork->mParams.linearOutput)
            mAdaptiveFilter[index]->Train(
                w.data(), std::atanh(mNetwork->mOut(index)),
                std::atanh(_value), mNetwork->mX.data());
        else
            mAdaptiveFilter[index]->Train(
                w.data(), mNetwork->mOut(index),
                _value, mNetwork->mX.data());

        mNetwork->mWOut.row(index) = w.transpose();

        if (force)
            mNetwork->mOut(index) = _value;
    }

    void TrainerImpl::TrainOnline(
        const std::vector<float> & output, bool forceOutput)
    {
        for (unsigned i = 0; i < mNetwork->mParams.outputCount; ++ i)
            TrainSingleOutputOnline(i, output[i], forceOutput);
    }


} // namespace ESN
