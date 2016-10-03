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

        RandomUniform(mWIn.data(),
            params.neuronCount * params.inputCount, -1.0f, 1.0f);

        // Generate weight matrix as random orthonormal matrix

        int neuronCountSqr = params.neuronCount * params.neuronCount;
        RandomUniform(mW.data(), neuronCountSqr, -1.0f, 1.0f);
        std::vector<float> conn(neuronCountSqr);
        RandomUniform(conn.data(), neuronCountSqr, 0.0f, 1.0f);
        for (int i = 0; i < neuronCountSqr; ++ i)
            if (conn[i] > params.connectivity)
                mW[i] = 0.0f;

        std::vector<float> s(params.neuronCount);
        std::vector<float> u(params.neuronCount * params.neuronCount);
        std::vector<float> vt(params.neuronCount * params.neuronCount);

        int info = SGESDD('A', params.neuronCount, params.neuronCount,
            mW.data(), params.neuronCount, s.data(), u.data(),
            params.neuronCount, vt.data(), params.neuronCount);
        if (info != 0)
            throw std::runtime_error("Failed to calculate SVD");

        SGEMM('N', 'T', params.neuronCount, params.neuronCount,
            params.neuronCount, 1.0f, u.data(), params.neuronCount,
            vt.data(), params.neuronCount, 0.0f, mW.data(),
            params.neuronCount);

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
        SGEMV('N', mParams.neuronCount, mParams.neuronCount, 1.0f,
            mW.data(), mParams.neuronCount, mX.data(), 1, 0.0f,
            mTemp.data(), 1);

        // mTemp = mWIn * mIn + mTemp
        SGEMV('N', mParams.neuronCount, mParams.inputCount, 1.0f,
            mWIn.data(), mParams.neuronCount, mIn.data(), 1, 1.0f,
            mTemp.data(), 1);

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
            SGEMV('N', mParams.neuronCount, mParams.outputCount, 1.0f,
                mWFB.data(), mParams.neuronCount, mOut.data(), 1, 1.0f,
                mTemp.data(), 1);
        }

        // mTemp[i] = tanh(mTemp[i])
        TanhEwise(mTemp.data(), mParams.neuronCount);

        // mX[i] *= mOneMinusLeakingRate[i]
        ProductEwise(mX.data(), mOneMinusLeakingRate.data(),
            mParams.neuronCount);

        // mX = mLeakingRate[i] * mTemp[i] + mX;
        SSBMV('L', mParams.neuronCount, 0, 1.0f, mLeakingRate.data(),
            1, mTemp.data(), 1, 1.0f, mX.data(), 1);

        // mOut = mWOut * mX
        SGEMV('N', mParams.outputCount, mParams.neuronCount, 1.0f,
            mWOut.data(), mParams.outputCount, mX.data(), 1, 0.0f,
            mOut.data(), 1);

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