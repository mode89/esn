// Copyright (c) 2016, Andrey Krainyak - All Rights Reserved
// You may use, distribute and modify this code under the terms of
// BSD 2-clause license.

#include <cmath>
#include <cstring>
#include <esn/exceptions.h>
#include <esn/math.h>
#include <esn/network_impl.h>

namespace ESN {

    std::shared_ptr<Network> CreateNetwork(
        const NetworkParams & params)
    {
        return std::make_shared<NetworkImpl>(params);
    }

    NetworkImpl::NetworkImpl( const NetworkParams & params )
        : mParams(params)
        , kOne(1.0f)
        , kMinusOne(-1.0f)
        , kZero(0.0f)
        , mIn(params.inputCount)
        , mWIn(params.neuronCount, params.inputCount)
        , mWInScaling(params.inputCount)
        , mWInBias(params.inputCount)
        , mX(params.neuronCount)
        , mW(params.neuronCount, params.neuronCount)
        , mLeakingRate(params.neuronCount)
        , mOneMinusLeakingRate(params.neuronCount)
        , mOut(params.outputCount)
        , mOutScale(params.outputCount)
        , mOutBias(params.outputCount)
        , mWOut(params.outputCount, params.neuronCount)
        , mWFB(params.neuronCount, params.outputCount)
        , mWFBScaling(params.outputCount)
        , mTemp(params.neuronCount)
        , mTempOut(params.outputCount)
    {
        if ( params.inputCount <= 0 )
            throw std::invalid_argument(
                "NetworkParams::inputCount must be not null" );
        if ( params.neuronCount <= 0 )
            throw std::invalid_argument(
                "NetworkParams::neuronCount must be not null" );
        if ( params.outputCount <= 0 )
            throw std::invalid_argument(
                "NetworkParams::outputCount must be not null" );
        if ( !( params.leakingRateMin > 0.0 &&
                params.leakingRateMin <= 1.0 ) )
            throw std::invalid_argument(
                "NetworkParams::leakingRateMin must be within "
                "interval (0,1]" );
        if ( !( params.leakingRateMax > 0.0 &&
                params.leakingRateMax <= 1.0 ) )
            throw std::invalid_argument(
                "NetworkParams::leakingRateMax must be within "
                "interval (0,1]" );
        if ( params.leakingRateMin > params.leakingRateMax )
            throw std::invalid_argument(
                "NetworkParams::leakingRateMin must be less then or "
                "equal to NetworkParams::leakingRateMax" );
        if ( !( params.connectivity > 0.0f &&
                params.connectivity <= 1.0f ) )
            throw std::invalid_argument(
                "NetworkParams::connectivity must be within "
                "interval (0,1]" );

        randm(kMinusOne, kOne, mWIn);

        // Generate weight matrix as random orthonormal matrix

        scalar<float> sparsity(1.0f - params.connectivity);
        randspm(kMinusOne, kOne, sparsity, mW);

        // Find S, U, VT from equation:
        // mW = U * S * VT
        vector<float> s(params.neuronCount);
        matrix<float> u(params.neuronCount, params.neuronCount);
        matrix<float> vt(params.neuronCount, params.neuronCount);
        if (gesvd('A', 'A', mW, s, u, vt) != 0)
            throw std::runtime_error("Failed to calculate SVD");

        // mW = U * VT
        gemm('N', 'N', kOne, u, vt, kZero, mW);

        fillv(kOne, mWInScaling);
        fillv(kZero, mWInBias);

        vector<float> vecWOut(mWOut.ptr(), mWOut.rows() * mWOut.cols());
        fillv(kZero, vecWOut);

        fillv(kOne, mOutScale);
        fillv(kZero, mOutBias);

        if (params.hasOutputFeedback)
        {
            randm(kMinusOne, kOne, mWFB);
            fillv(kOne, mWFBScaling);
        }

        scalar<float> leakingRateMin(params.leakingRateMin);
        scalar<float> leakingRateMax(params.leakingRateMax);
        randv(leakingRateMin, leakingRateMax, mLeakingRate);

        // mOneMinusLeakingRate[i] = 1.0f - mLeakingRate[i]
        fillv(kOne, mOneMinusLeakingRate);
        axpy(kMinusOne, mLeakingRate, mOneMinusLeakingRate);

        fillv(kZero, mIn);
        randv(kMinusOne, kOne, mX);
        fillv(kZero, mOut);
    }

    NetworkImpl::~NetworkImpl()
    {
    }

    void NetworkImpl::SetInputs( const std::vector< float > & inputs )
    {
        if (inputs.size() != mParams.inputCount)
            throw std::invalid_argument( "Wrong size of the input vector" );

        mIn = inputs;
        axpy(kOne, mWInBias, mIn);
        prodvv(mWInScaling, mIn);
    }

    void NetworkImpl::SetInputScalings(
        const std::vector< float > & scalings )
    {
        if ( scalings.size() != mParams.inputCount )
            throw std::invalid_argument(
                "Wrong size of the scalings vector" );

        mWInScaling = scalings;
    }

    void NetworkImpl::SetInputBias(
        const std::vector< float > & bias )
    {
        if ( bias.size() != mParams.inputCount )
            throw std::invalid_argument(
                "Wrong size of the scalings vector" );

        mWInBias = bias;
    }

    void NetworkImpl::SetOutputScale(const std::vector<float> & scale)
    {
        if (scale.size() != mParams.outputCount)
            throw std::invalid_argument(
                "Wrong size of the output scale vector");

        mOutScale = scale;
    }

    void NetworkImpl::SetOutputBias(const std::vector<float> & bias)
    {
        if (bias.size() != mParams.outputCount)
            throw std::invalid_argument(
                "Wrong size of the output bias vector");

        mOutBias = bias;
    }

    void NetworkImpl::SetFeedbackScalings(
        const std::vector< float > & scalings )
    {
        if (!mParams.hasOutputFeedback)
            throw std::logic_error(
                "Trying to set up feedback scaling for a network "
                "which doesn't have an output feedback");
        if ( scalings.size() != mParams.outputCount )
            throw std::invalid_argument(
                "Wrong size of the scalings vector" );

        mWFBScaling = scalings;
    }

    void NetworkImpl::Step( float step )
    {
        if ( step <= 0.0f )
            throw std::invalid_argument(
                "Step size must be positive value" );

        // mTemp = mW * mX
        gemv('N', kOne, mW, mX, kZero, mTemp);

        // mTemp = mWIn * mIn + mTemp
        gemv('N', kOne, mWIn, mIn, kOne, mTemp);

        if (mParams.hasOutputFeedback)
        {
            if (mParams.linearOutput)
            {
                // mOut[i] = tanh(mOut[i])
                tanhv(mOut);
            }

            // mOut[i] *= mWFBScaling[i]
            prodvv(mWFBScaling, mOut);

            // mTemp = mWFB * mOut + mTemp
            gemv('N', kOne, mWFB, mOut, kOne, mTemp);
        }

        // mTemp[i] = tanh(mTemp[i])
        tanhv(mTemp);

        // mX[i] *= mOneMinusLeakingRate[i]
        prodvv(mOneMinusLeakingRate, mX);

        // mX = mLeakingRate[i] * mTemp[i] + mX;
        sbmv('L', mParams.neuronCount, 0, kOne, mLeakingRate, 1, mTemp,
            kOne, mX);

        // mOut = mWOut * mX
        gemv('N', kOne, mWOut, mX, kZero, mOut);

        if (!mParams.linearOutput)
            // mOut[i] = tanh(mOut[i])
            tanhv(mOut);

        for (int i = 0; i < mParams.outputCount; ++ i)
            if (!std::isfinite(static_cast<float>(mOut[i])))
                throw OutputIsNotFinite();
    }

    void NetworkImpl::CaptureTransformedInput(
        std::vector< float > & input )
    {
        if ( input.size() != mParams.inputCount )
            throw std::invalid_argument(
                "Size of the vector must be equal to "
                "the number of inputs" );

        input = mIn;
    }

    void NetworkImpl::CaptureActivations(
        std::vector< float > & activations )
    {
        if ( activations.size() != mParams.neuronCount )
            throw std::invalid_argument(
                "Size of the vector must be equal "
                "actual number of neurons" );

        activations = mX;
    }

    void NetworkImpl::CaptureOutput( std::vector< float > & output )
    {
        if ( output.size() != mParams.outputCount )
            throw std::invalid_argument(
                "Size of the vector must be equal "
                "actual number of outputs" );

        // output[i] = mOut[i] * mOutScale[i] + mOutBias[i]
        copy(mOutBias, mTempOut);
        sbmv('L', mParams.outputCount, 0, kOne, mOut, 1, mOutScale,
            kOne, mTempOut);
        output = mTempOut;
    }

} // namespace ESN
