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
        , kMinusOne(-1.0f)
        , mNetwork(std::static_pointer_cast<NetworkImpl>(network))
        , mAdaptiveFilter(mNetwork->mParams.outputCount)
        , mTempValue(0.0f)
        , mTempAtanhOut(0.0f)
        , mTempAtanhValue(0.0f)
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
        mTempValue = value;
        axpy(kMinusOne, mNetwork->mOutBias[index], mTempValue);
        divvv(mTempValue, mNetwork->mOutScale[index]);

        // Extract row of weights corresponding to the output
        vector<float> w = mNetwork->mWOut[index];

        if (!mNetwork->mParams.linearOutput)
        {
            copy(mNetwork->mOut[index], mTempAtanhOut);
            atanhv(mTempAtanhOut);
            copy(mTempValue, mTempAtanhValue);
            atanhv(mTempAtanhValue);
            mAdaptiveFilter[index]->Train(
                w, mTempAtanhOut, mTempAtanhValue, mNetwork->mX.ptr());
        }
        else
            mAdaptiveFilter[index]->Train(
                w, mNetwork->mOut[index], mTempValue, mNetwork->mX.ptr());

        if (force)
        {
            scalar<float> out(mNetwork->mOut[index]);
            copy(mTempValue, out);
        }
    }

    void TrainerImpl::TrainOnline(
        const std::vector<float> & output, bool forceOutput)
    {
        for (unsigned i = 0; i < mNetwork->mParams.outputCount; ++ i)
            TrainSingleOutputOnline(i, output[i], forceOutput);
    }


} // namespace ESN
