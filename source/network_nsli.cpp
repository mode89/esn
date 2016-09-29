#include <cmath>
#include <cstring>
#include <Eigen/Eigenvalues>
#include <esn/exceptions.h>
#include <esn/network_nsli.h>
#include <network_nsli.h>
#include <random>

extern "C" {
#include <cblas/cblas.h>
#include <clapack/clapack.h>
}

std::default_random_engine sRandomEngine;

void RandomUniform(float * v, int size, float a, float b)
{
    std::uniform_real_distribution<float> dist(a, b);
    for (int i = 0; i < size; ++ i)
        v[i] = dist(sRandomEngine);
}

void Constant(float * v, int size, float value)
{
    for (int i = 0; i < size; ++ i)
        v[i] = value;
}

namespace ESN {

    std::shared_ptr<Network> CreateNetwork(
        const NetworkParamsNSLI & params)
    {
        return std::shared_ptr<NetworkNSLI>(new NetworkNSLI(params));
    }

    NetworkNSLI::NetworkNSLI( const NetworkParamsNSLI & params )
        : mParams( params )
        , mIn( params.inputCount )
        , mWIn( params.neuronCount, params.inputCount )
        , mWInScaling( params.inputCount )
        , mWInBias( params.inputCount )
        , mX( params.neuronCount )
        , mW( params.neuronCount, params.neuronCount )
        , mLeakingRate(params.neuronCount)
        , mOut( params.outputCount )
        , mOutScale(params.outputCount)
        , mOutBias(params.outputCount)
        , mWOut( params.outputCount, params.neuronCount )
        , mWFB(params.neuronCount, params.outputCount)
        , mWFBScaling(params.outputCount)
        , mAdaptiveFilter(params.outputCount)
    {
        if ( params.inputCount <= 0 )
            throw std::invalid_argument(
                "NetworkParamsNSLI::inputCount must be not null" );
        if ( params.neuronCount <= 0 )
            throw std::invalid_argument(
                "NetworkParamsNSLI::neuronCount must be not null" );
        if ( params.outputCount <= 0 )
            throw std::invalid_argument(
                "NetworkParamsNSLI::outputCount must be not null" );
        if ( !( params.leakingRateMin > 0.0 &&
                params.leakingRateMin <= 1.0 ) )
            throw std::invalid_argument(
                "NetworkParamsNSLI::leakingRateMin must be within "
                "interval (0,1]" );
        if ( !( params.leakingRateMax > 0.0 &&
                params.leakingRateMax <= 1.0 ) )
            throw std::invalid_argument(
                "NetworkParamsNSLI::leakingRateMax must be within "
                "interval (0,1]" );
        if ( params.leakingRateMin > params.leakingRateMax )
            throw std::invalid_argument(
                "NetworkParamsNSLI::leakingRateMin must be less then or "
                "equal to NetworkParamsNSLI::leakingRateMax" );
        if ( !( params.connectivity > 0.0f &&
                params.connectivity <= 1.0f ) )
            throw std::invalid_argument(
                "NetworkParamsNSLI::connectivity must be within "
                "interval (0,1]" );

        RandomUniform(mWIn.data(),
            params.neuronCount * params.inputCount, -1.0f, 1.0f);

        int neuronCountSqr = params.neuronCount * params.neuronCount;
        Eigen::MatrixXf randomWeights(
            params.neuronCount, params.neuronCount);
        RandomUniform(randomWeights.data(), neuronCountSqr, -1.0f, 1.0f);
        std::uniform_real_distribution<float> uniDist;
        for (int i = 0; i < neuronCountSqr; ++ i)
            if (uniDist(sRandomEngine) > params.connectivity)
                randomWeights.data()[i] = 0.0f;

        if ( params.useOrthonormalMatrix )
        {
            int n = params.neuronCount;
            float * A = randomWeights.data();

            char job = 'A';
            std::vector<float> s(n);
            std::vector<float> u(n * n);
            std::vector<float> vt(n * n);
            int lwork = 5 * n;
            std::vector<float> work(lwork);
            int info = 0;

            sgesvd_(&job, &job, &n, &n, A, &n, s.data(), u.data(), &n,
                vt.data(), &n, work.data(), &lwork, &info);
            if (info != 0)
                throw std::runtime_error("Failed to calculate SVD");

            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n,
                1.0f, u.data(), n, vt.data(), n, 0.0f, mW.data(), n);
        }
        else
        {
            float spectralRadius =
                randomWeights.eigenvalues().cwiseAbs().maxCoeff();
            mW = randomWeights / spectralRadius * params.spectralRadius;
        }

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
        mOneMinusLeakingRate = 1.0f - mLeakingRate.array();

        Constant(mIn.data(), params.inputCount, 0.0f);
        RandomUniform(mX.data(), params.neuronCount, -1.0f, 1.0f);
        Constant(mOut.data(), params.outputCount, 0.0f);

        for (int i = 0; i < params.outputCount; ++i)
            mAdaptiveFilter[i] = std::make_shared<AdaptiveFilterRLS>(
                params.neuronCount,
                params.onlineTrainingForgettingFactor,
                params.onlineTrainingInitialCovariance);
    }

    NetworkNSLI::~NetworkNSLI()
    {
    }

    void NetworkNSLI::SetInputs( const std::vector< float > & inputs )
    {
        if ( inputs.size() != mIn.rows() )
            throw std::invalid_argument( "Wrong size of the input vector" );
        mIn = (Eigen::Map<Eigen::VectorXf>(
            const_cast<float*>(inputs.data()), inputs.size()) +
            mWInBias).cwiseProduct(mWInScaling);
    }

    void NetworkNSLI::SetInputScalings(
        const std::vector< float > & scalings )
    {
        if ( scalings.size() != mParams.inputCount )
            throw std::invalid_argument(
                "Wrong size of the scalings vector" );
        mWInScaling = Eigen::Map< Eigen::VectorXf >(
            const_cast< float * >( scalings.data() ), scalings.size() );
    }

    void NetworkNSLI::SetInputBias(
        const std::vector< float > & bias )
    {
        if ( bias.size() != mParams.inputCount )
            throw std::invalid_argument(
                "Wrong size of the scalings vector" );
        mWInBias = Eigen::Map< Eigen::VectorXf >(
            const_cast< float * >( bias.data() ), bias.size() );
    }

    void NetworkNSLI::SetOutputScale(const std::vector<float> & scale)
    {
        if (scale.size() != mParams.outputCount)
            throw std::invalid_argument(
                "Wrong size of the output scale vector");
        mOutScale = Eigen::Map<Eigen::VectorXf>(
            const_cast<float*>(scale.data()), scale.size());
    }

    void NetworkNSLI::SetOutputBias(const std::vector<float> & bias)
    {
        if (bias.size() != mParams.outputCount)
            throw std::invalid_argument(
                "Wrong size of the output bias vector");
        mOutBias = Eigen::Map<Eigen::VectorXf>(
            const_cast<float*>(bias.data()), bias.size());
    }

    void NetworkNSLI::SetFeedbackScalings(
        const std::vector< float > & scalings )
    {
        if (!mParams.hasOutputFeedback)
            throw std::logic_error(
                "Trying to set up feedback scaling for a network "
                "which doesn't have an output feedback");
        if ( scalings.size() != mParams.outputCount )
            throw std::invalid_argument(
                "Wrong size of the scalings vector" );
        mWFBScaling = Eigen::Map< Eigen::VectorXf >(
            const_cast< float * >( scalings.data() ), scalings.size() );
    }

    void NetworkNSLI::Step( float step )
    {
        if ( step <= 0.0f )
            throw std::invalid_argument(
                "Step size must be positive value" );

        auto tanh = [] ( float x ) -> float { return std::tanh( x ); };

        #define TEMP mWIn * mIn + mW * mX

        #define CALC_X(val) \
            mX = mOneMinusLeakingRate.cwiseProduct(mX) + \
                mLeakingRate.cwiseProduct((val).unaryExpr(tanh))

        #define CALC_X_WITH_FB(val) \
            if (mParams.hasOutputFeedback) \
                CALC_X(TEMP + val); \
            else \
                CALC_X(TEMP)

        if ( mParams.linearOutput )
        {
            CALC_X_WITH_FB(
                mWFB * mOut.unaryExpr(tanh).cwiseProduct(mWFBScaling));
            mOut = mWOut * mX;
        }
        else
        {
            CALC_X_WITH_FB(
                mWFB * mOut.cwiseProduct(mWFBScaling));
            mOut = ( mWOut * mX ).unaryExpr( tanh );
        }

        #undef TEMP
        #undef CALC_X
        #undef CALC_X_WITH_FB

        auto isnotfinite =
            [] (float n) -> bool { return !std::isfinite(n); };
        if (mOut.unaryExpr(isnotfinite).any())
            throw OutputIsNotFinite();
    }

    void NetworkNSLI::CaptureTransformedInput(
        std::vector< float > & input )
    {
        if ( input.size() != mParams.inputCount )
            throw std::invalid_argument(
                "Size of the vector must be equal to "
                "the number of inputs" );
        for ( int i = 0; i < mParams.inputCount; ++ i )
            input[ i ] = mIn( i );
    }

    void NetworkNSLI::CaptureActivations(
        std::vector< float > & activations )
    {
        if ( activations.size() != mParams.neuronCount )
            throw std::invalid_argument(
                "Size of the vector must be equal "
                "actual number of neurons" );

        for ( int i = 0; i < mParams.neuronCount; ++ i )
            activations[ i ] = mX( i );
    }

    void NetworkNSLI::CaptureOutput( std::vector< float > & output )
    {
        if ( output.size() != mParams.outputCount )
            throw std::invalid_argument(
                "Size of the vector must be equal "
                "actual number of outputs" );

        for ( int i = 0; i < mParams.outputCount; ++ i )
            output[i] = mOut(i) * mOutScale(i) + mOutBias(i);
    }

    void NetworkNSLI::Train(
        const std::vector< std::vector< float > > & inputs,
        const std::vector< std::vector< float > > & outputs )
    {
        if ( inputs.size() == 0 )
            throw std::invalid_argument(
                "Number of samples must be not null" );
        if ( inputs.size() != outputs.size() )
            throw std::invalid_argument(
                "Number of input and output samples must be equal" );
        const unsigned kSampleCount = inputs.size();

        Eigen::MatrixXf matX( mParams.neuronCount, kSampleCount );
        Eigen::MatrixXf matY( mParams.outputCount, kSampleCount );
        for ( int i = 0; i < kSampleCount; ++ i )
        {
            SetInputs( inputs[i] );
            Step( 0.1f );
            matX.col( i ) = mX;
            matY.col( i ) = Eigen::Map< Eigen::VectorXf >(
                const_cast< float * >( outputs[i].data() ),
                    mParams.outputCount );
        }

        Eigen::MatrixXf matXT = matX.transpose();

        mWOut = ( matY * matXT * ( matX * matXT ).inverse() );
    }

    void NetworkNSLI::TrainSingleOutputOnline(
        unsigned index, float value, bool force)
    {
        // Calculate output without bias and scaling
        float _value = (value - mOutBias(index)) / mOutScale(index);

        Eigen::VectorXf w = mWOut.row(index).transpose();
        if (!mParams.linearOutput)
            mAdaptiveFilter[index]->Train(w, std::atanh(mOut(index)),
                std::atanh(_value), mX);
        else
            mAdaptiveFilter[index]->Train(w, mOut(index), _value, mX);

        mWOut.row(index) = w.transpose();

        if (force)
            mOut(index) = _value;
    }

    void NetworkNSLI::TrainOnline(
        const std::vector<float> & output, bool forceOutput)
    {
        for (unsigned i = 0; i < mParams.outputCount; ++ i)
            TrainSingleOutputOnline(i, output[i], forceOutput);
    }

} // namespace ESN
