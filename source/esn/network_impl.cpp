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
        , mWOut(params.outputCount * params.neuronCount)
        , mWFB(params.neuronCount, params.outputCount)
        , mWFBScaling(params.outputCount)
        , mTemp(params.neuronCount)
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

        srandv(params.neuronCount * params.inputCount,
            kMinusOne.ptr(), kOne.ptr(), mWIn.ptr());

        // Generate weight matrix as random orthonormal matrix

        int neuronCountSqr = params.neuronCount * params.neuronCount;
        scalar<float> sparsity(1.0f - params.connectivity);
        srandspv(neuronCountSqr,
            kMinusOne.ptr(), kOne.ptr(), sparsity.ptr(), mW.ptr());

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

        Constant(mWOut.data(),
            params.outputCount * params.neuronCount, 0.0f);

        Constant(mOutScale.data(), params.outputCount, 1.0f);
        Constant(mOutBias.data(), params.outputCount, 0.0f);

        if (params.hasOutputFeedback)
        {
            srandv(params.neuronCount * params.outputCount,
                kMinusOne.ptr(), kOne.ptr(), mWFB.ptr());
            fillv(kOne, mWFBScaling);
        }

        scalar<float> leakingRateMin(params.leakingRateMin);
        scalar<float> leakingRateMax(params.leakingRateMax);
        srandv(params.neuronCount, leakingRateMin.ptr(),
            leakingRateMax.ptr(), mLeakingRate.ptr());

        // mOneMinusLeakingRate[i] = 1.0f - mLeakingRate[i]
        fillv(kOne, mOneMinusLeakingRate);
        axpy(kMinusOne, mLeakingRate, mOneMinusLeakingRate);

        fillv(kZero, mIn);
        srandv(params.neuronCount, kMinusOne.ptr(), kOne.ptr(), mX.ptr());
        Constant(mOut.data(), params.outputCount, 0.0f);
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
        sprodvv(mParams.inputCount, mWInScaling.ptr(), mIn.ptr());
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

        SCOPY(mParams.outputCount, scale.data(), 1, mOutScale.data(), 1);
    }

    void NetworkImpl::SetOutputBias(const std::vector<float> & bias)
    {
        if (bias.size() != mParams.outputCount)
            throw std::invalid_argument(
                "Wrong size of the output bias vector");

        SCOPY(mParams.outputCount, bias.data(), 1, mOutBias.data(), 1);
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
        sgemv('N', mParams.neuronCount, mParams.neuronCount, kOne.ptr(),
            mW.ptr(), mParams.neuronCount, mX.ptr(), 1, kZero.ptr(),
            mTemp.ptr(), 1);
        // SGEMV('N', mParams.neuronCount, mParams.neuronCount, 1.0f,
        //     mW.data(), mParams.neuronCount, mX.data(), 1, 0.0f,
        //     mTemp.data(), 1);

        // mTemp = mWIn * mIn + mTemp
        sgemv('N', mParams.neuronCount, mParams.inputCount, kOne.ptr(),
            mWIn.ptr(), mParams.neuronCount, mIn.ptr(), 1, kOne.ptr(),
            mTemp.ptr(), 1);
        // SGEMV('N', mParams.neuronCount, mParams.inputCount, 1.0f,
        //     mWIn.data(), mParams.neuronCount, mIn.data(), 1, 1.0f,
        //     mTemp.data(), 1);

        if (mParams.hasOutputFeedback)
        {
            if (mParams.linearOutput)
            {
                // mOut[i] = tanh(mOut[i])
                TanhEwise(mOut.data(), mParams.outputCount);
            }

            // mOut[i] *= mWFBScaling[i]
            vector<float> vecOut(mOut);
            sprodvv(mParams.outputCount, mWFBScaling.ptr(), vecOut.ptr());

            // mTemp = mWFB * mOut + mTemp
            sgemv('N', mParams.neuronCount, mParams.outputCount, kOne.ptr(),
                mWFB.ptr(), mParams.neuronCount, vecOut.ptr(), 1, kOne.ptr(),
                mTemp.ptr(), 1);
            // SGEMV('N', mParams.neuronCount, mParams.outputCount, 1.0f,
            //     mWFB.data(), mParams.neuronCount, mOut.data(), 1, 1.0f,
            //     mTemp.data(), 1);
        }

        // mTemp[i] = tanh(mTemp[i])
        stanhv(mParams.neuronCount, mTemp.ptr());

        // mX[i] *= mOneMinusLeakingRate[i]
        sprodvv(mParams.neuronCount, mOneMinusLeakingRate.ptr(), mX.ptr());

        // mX = mLeakingRate[i] * mTemp[i] + mX;
        ssbmv('L', mParams.neuronCount, 0, kOne.ptr(), mLeakingRate.ptr(),
            1, mTemp.ptr(), 1, kOne.ptr(), mX.ptr(), 1);
        // SSBMV('L', mParams.neuronCount, 0, 1.0f, mLeakingRate.data(),
        //     1, mTemp.data(), 1, 1.0f, mX.data(), 1);

        // mOut = mWOut * mX
        matrix<float> matWOut(mWOut,
            mParams.outputCount, mParams.neuronCount);
        vector<float> vecOut(mOut);
        sgemv('N', mParams.outputCount, mParams.neuronCount, kOne.ptr(),
            matWOut.ptr(), mParams.outputCount, mX.ptr(), 1, kZero.ptr(),
            vecOut.ptr(), 1);
        mOut = vecOut;
        // SGEMV('N', mParams.outputCount, mParams.neuronCount, 1.0f,
        //     mWOut.data(), mParams.outputCount, mX.data(), 1, 0.0f,
        //     mOut.data(), 1);

        if (!mParams.linearOutput)
            // mOut[i] = tanh(mOut[i])
            TanhEwise(mOut.data(), mParams.outputCount);

        for (int i = 0; i < mParams.outputCount; ++ i)
            if (!std::isfinite(mOut[i]))
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

        for (int i = 0; i < mParams.outputCount; ++ i)
            output[i] = mOut[i] * mOutScale[i] + mOutBias[i];
    }

} // namespace ESN
