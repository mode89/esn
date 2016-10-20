// Copyright (c) 2016, Andrey Krainyak - All Rights Reserved
// You may use, distribute and modify this code under the terms of
// BSD 2-clause license.

#include <cmath>
#include <esn/adaptive_filter_rls.h>
#include <esn/math.h>
#include <esn/network_impl.h>
#include <esn/trainer_impl.h>

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
        , mNetwork(std::static_pointer_cast<NetworkImpl>(network))
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
        float _value = (value -
            static_cast<float>(mNetwork->mOutBias[index])) /
                static_cast<float>(mNetwork->mOutScale[index]);

        // Extract row of weights corresponding to the output
        const int neuronCount = mNetwork->mParams.neuronCount;
        const int outputCount = mNetwork->mParams.outputCount;
        std::vector<float> w(neuronCount);
        SCOPY(neuronCount, &mNetwork->mWOut[index],
            outputCount, w.data(), 1);

        if (!mNetwork->mParams.linearOutput)
            mAdaptiveFilter[index]->Train(
                w.data(), std::atanh(mNetwork->mOut[index]),
                std::atanh(_value), mNetwork->mX.ptr());
        else
            mAdaptiveFilter[index]->Train(
                w.data(), mNetwork->mOut[index],
                _value, mNetwork->mX.ptr());

        // Write back the row of weights
        SCOPY(neuronCount, w.data(), 1,
            &mNetwork->mWOut[index], outputCount);

        if (force)
            mNetwork->mOut[index] = _value;
    }

    void TrainerImpl::TrainOnline(
        const std::vector<float> & output, bool forceOutput)
    {
        for (unsigned i = 0; i < mNetwork->mParams.outputCount; ++ i)
            TrainSingleOutputOnline(i, output[i], forceOutput);
    }


} // namespace ESN
