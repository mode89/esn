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
        , kOne(make_pointer(1.0f))
        , kMinusOne(make_pointer(-1.0f))
        , kZero(make_pointer(0.0f))
        , mIn(params.inputCount)
        , mWIn(params.neuronCount * params.inputCount)
        , mWInScaling(params.inputCount)
        , mWInBias(params.inputCount)
        , mX(params.neuronCount)
        , mW(params.neuronCount * params.neuronCount)
        , mLeakingRate(params.neuronCount)
        , mOneMinusLeakingRate(params.neuronCount)
        , mOut(params.outputCount)
        , mOutScale(params.outputCount)
        , mOutBias(params.outputCount)
        , mWOut(params.outputCount * params.neuronCount)
        , mWFB(params.neuronCount * params.outputCount)
        , mWFBScaling(params.outputCount)
        , mTemp(make_pointer(params.neuronCount * sizeof(float)))
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

        RandomUniform(mWIn.data(),
            params.neuronCount * params.inputCount, -1.0f, 1.0f);

        // Generate weight matrix as random orthonormal matrix

        int neuronCountSqr = params.neuronCount * params.neuronCount;
        pointer ptrW = make_pointer(mW);
        pointer ptrMinusOne = make_pointer(-1.0f);
        pointer ptrOne = make_pointer(1.0f);
        pointer ptrSparsity = make_pointer(1.0f - params.connectivity);
        srandspv(neuronCountSqr, ptrMinusOne, ptrOne, ptrSparsity, ptrW);
        memcpy(mW, ptrW);

        // Find S, U, VT from equation:
        // mW = U * S * VT
        pointer ptrA = make_pointer(mW);
        pointer ptrS = make_pointer(params.neuronCount * sizeof(float));
        pointer ptrU = make_pointer(
            params.neuronCount * params.neuronCount * sizeof(float));
        pointer ptrVT = make_pointer(
            params.neuronCount * params.neuronCount * sizeof(float));
        int info = sgesvd('A', 'A', params.neuronCount, params.neuronCount,
            ptrA, params.neuronCount, ptrS, ptrU, params.neuronCount, ptrVT,
            params.neuronCount);
        // int info = SGESDD('A', params.neuronCount, params.neuronCount,
        //     mW.data(), params.neuronCount, s.data(), u.data(),
        //     params.neuronCount, vt.data(), params.neuronCount);
        if (info != 0)
            throw std::runtime_error("Failed to calculate SVD");

        // mW = U * VT
        pointer ptrAlpha = make_pointer(1.0f);
        pointer ptrBeta = make_pointer(0.0f);
        sgemm('N', 'N', params.neuronCount, params.neuronCount,
            params.neuronCount, ptrAlpha, ptrU, params.neuronCount,
            ptrVT, params.neuronCount, ptrBeta, ptrW, params.neuronCount);
        memcpy(mW, ptrW);
        // SGEMM('N', 'N', params.neuronCount, params.neuronCount,
        //     params.neuronCount, 1.0f, u.data(), params.neuronCount,
        //     vt.data(), params.neuronCount, 0.0f, mW.data(),
        //     params.neuronCount);

        Constant(mWInScaling.data(), params.inputCount, 1.0f);
        Constant(mWInBias.data(), params.inputCount, 0.0f);

        Constant(mWOut.data(),
            params.outputCount * params.neuronCount, 0.0f);

        Constant(mOutScale.data(), params.outputCount, 1.0f);
        Constant(mOutBias.data(), params.outputCount, 0.0f);

        if (params.hasOutputFeedback)
        {
            RandomUniform(mWFB.data(),
                params.neuronCount * params.outputCount, -1.0f, 1.0f);
            Constant(mWFBScaling.data(), params.outputCount, 1.0f);
        }

        RandomUniform(mLeakingRate.data(), params.neuronCount,
            params.leakingRateMin, params.leakingRateMax);
        // mOneMinusLeakingRate[i] = 1.0f - mLeakingRate[i]
        Constant(mOneMinusLeakingRate.data(), params.neuronCount, 1.0f);
        SAXPY(params.neuronCount, -1.0f, mLeakingRate.data(), 1,
            mOneMinusLeakingRate.data(), 1);

        Constant(mIn.data(), params.inputCount, 0.0f);
        RandomUniform(mX.data(), params.neuronCount, -1.0f, 1.0f);
        Constant(mOut.data(), params.outputCount, 0.0f);
    }

    NetworkImpl::~NetworkImpl()
    {
    }

    void NetworkImpl::SetInputs( const std::vector< float > & inputs )
    {
        if (inputs.size() != mParams.inputCount)
            throw std::invalid_argument( "Wrong size of the input vector" );

        SumEwise(mIn.data(), inputs.data(), mWInBias.data(), inputs.size());
        ProductEwise(mIn.data(), mWInScaling.data(), inputs.size());
    }

    void NetworkImpl::SetInputScalings(
        const std::vector< float > & scalings )
    {
        if ( scalings.size() != mParams.inputCount )
            throw std::invalid_argument(
                "Wrong size of the scalings vector" );

        SCOPY(mParams.inputCount, scalings.data(), 1,
            mWInScaling.data(), 1);
    }

    void NetworkImpl::SetInputBias(
        const std::vector< float > & bias )
    {
        if ( bias.size() != mParams.inputCount )
            throw std::invalid_argument(
                "Wrong size of the scalings vector" );

        SCOPY(mParams.inputCount, bias.data(), 1, mWInBias.data(), 1);
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

        SCOPY(mParams.outputCount, scalings.data(), 1,
            mWFBScaling.data(), 1);
    }

    void NetworkImpl::Step( float step )
    {
        if ( step <= 0.0f )
            throw std::invalid_argument(
                "Step size must be positive value" );

        // mTemp = mW * mX
        pointer ptrAlpha = make_pointer(1.0f);
        pointer ptrW = make_pointer(mW);
        pointer ptrX = make_pointer(mX);
        pointer ptrBeta = make_pointer(0.0f);
        sgemv('N', mParams.neuronCount, mParams.neuronCount, ptrAlpha,
            ptrW, mParams.neuronCount, ptrX, 1, ptrBeta, mTemp, 1);
        // SGEMV('N', mParams.neuronCount, mParams.neuronCount, 1.0f,
        //     mW.data(), mParams.neuronCount, mX.data(), 1, 0.0f,
        //     mTemp.data(), 1);

        // mTemp = mWIn * mIn + mTemp
        memcpy(ptrAlpha, 1.0f);
        pointer ptrWIn = make_pointer(mWIn);
        pointer ptrIn = make_pointer(mIn);
        memcpy(ptrBeta, 1.0f);
        sgemv('N', mParams.neuronCount, mParams.inputCount, ptrAlpha,
            ptrWIn, mParams.neuronCount, ptrIn, 1, ptrBeta, mTemp, 1);
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
            ProductEwise(mOut.data(), mWFBScaling.data(),
                mParams.outputCount);

            // mTemp = mWFB * mOut + mTemp
            memcpy(ptrAlpha, 1.0f);
            pointer ptrWFB = make_pointer(mWFB);
            pointer ptrOut = make_pointer(mOut);
            memcpy(ptrBeta, 1.0f);
            sgemv('N', mParams.neuronCount, mParams.outputCount, ptrAlpha,
                ptrWFB, mParams.neuronCount, ptrOut, 1, ptrBeta,
                mTemp, 1);
            // SGEMV('N', mParams.neuronCount, mParams.outputCount, 1.0f,
            //     mWFB.data(), mParams.neuronCount, mOut.data(), 1, 1.0f,
            //     mTemp.data(), 1);
        }

        // mTemp[i] = tanh(mTemp[i])
        stanhv(mParams.neuronCount, mTemp);

        // mX[i] *= mOneMinusLeakingRate[i]
        ProductEwise(mX.data(), mOneMinusLeakingRate.data(),
            mParams.neuronCount);

        // mX = mLeakingRate[i] * mTemp[i] + mX;
        memcpy(ptrAlpha, 1.0f);
        pointer ptrLeakingRate = make_pointer(mLeakingRate);
        memcpy(ptrBeta, 1.0f);
        memcpy(ptrX, mX);
        ssbmv('L', mParams.neuronCount, 0, ptrAlpha, ptrLeakingRate, 1,
            mTemp, 1, ptrBeta, ptrX, 1);
        memcpy(mX, ptrX);
        // SSBMV('L', mParams.neuronCount, 0, 1.0f, mLeakingRate.data(),
        //     1, mTemp.data(), 1, 1.0f, mX.data(), 1);

        // mOut = mWOut * mX
        memcpy(ptrAlpha, 1.0f);
        pointer ptrWOut = make_pointer(mWOut);
        memcpy(ptrX, mX);
        memcpy(ptrBeta, 0.0f);
        pointer ptrOut = make_pointer(mOut);
        sgemv('N', mParams.outputCount, mParams.neuronCount, ptrAlpha,
            ptrWOut, mParams.outputCount, ptrX, 1, ptrBeta, ptrOut, 1);
        memcpy(mOut, ptrOut);
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

        SCOPY(mParams.inputCount, mIn.data(), 1, input.data(), 1);
    }

    void NetworkImpl::CaptureActivations(
        std::vector< float > & activations )
    {
        if ( activations.size() != mParams.neuronCount )
            throw std::invalid_argument(
                "Size of the vector must be equal "
                "actual number of neurons" );

        SCOPY(mParams.neuronCount, mX.data(), 1, activations.data(), 1);
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
