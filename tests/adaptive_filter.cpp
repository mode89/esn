#include <cmath>
#include <esn/adaptive_filter_rls.h>
#include <esn/math.h>
#include <gtest/gtest.h>

const unsigned kInputCount = 100;
const float kMaxAmplitude = 1.0f;
const float kMaxFrequency = 10.0f;
const float kStep = 0.1f * 1.0f / kMaxFrequency;

class ReferenceFilter
{
public:
    ReferenceFilter( unsigned inputCount )
        : mW(inputCount)
    {
        ESN::RandomUniform(mW.data(), inputCount, -1.0f, 1.0f);
    }

    float operator()(const std::vector<float> input)
    {
        return ESN::SDOT(input.size(), mW.data(), 1, input.data(), 1);
    }

    std::vector<float> mW;
};

class Model
{
public:
    Model()
        : mAmplitude(kInputCount)
        , mOmega(kInputCount)
        , mInput(kInputCount)
        , mW(kInputCount)
        , mOutput(0.0f)
        , mReferenceFilter(kInputCount)
        , mReferenceOutput(0.0f)
        , mTime(0.0f)
    {
        ESN::RandomUniform(mAmplitude.data(), kInputCount, 0.0f,
            kMaxAmplitude);
        ESN::RandomUniform(mOmega.data(), kInputCount, 0.0f, kMaxFrequency);
        ESN::RandomUniform(mW.data(), kInputCount, -1.0f, 1.0f);
    }

    void Update()
    {
        mTime += kStep;

        for (int i = 0; i < kInputCount; ++ i)
            mInput[i] = mAmplitude[i] * std::sin(mOmega[i] * mTime);

        mOutput = ESN::SDOT(kInputCount, mW.data(), 1, mInput.data(), 1);
        mReferenceOutput = mReferenceFilter(mInput);
    }

    std::vector<float> mAmplitude;
    std::vector<float> mOmega;
    std::vector<float> mInput;
    std::vector<float> mW;
    float mOutput;
    ReferenceFilter mReferenceFilter;
    float mReferenceOutput;
    float mTime;
};

TEST( AdaptiveFilter, NLMS )
{
    const unsigned kStepCount = 10000;
    const float kTrainStep = 0.1f;

    Model model;
    model.Update();
    float error = model.mReferenceOutput - model.mOutput;
    float initialError = std::fabs( error / model.mOutput );

    for ( int i = 0; i < kStepCount; ++ i )
    {
        model.Update();
        error = model.mReferenceOutput - model.mOutput;

        float inputNorm = 0.0f;
        for (int j = 0; j < kInputCount; ++ j)
        {
            float input = model.mInput[j];
            inputNorm += input * input;
        }
        inputNorm = std::sqrt(inputNorm);

        for (int j = 0; j < kInputCount; ++ j)
            model.mW[j] +=
                (kTrainStep * error * model.mInput[j] / inputNorm);
    }

    EXPECT_TRUE( std::fabs( error / model.mOutput ) < initialError );
}

TEST(AdaptiveFilter, RLS)
{
    const unsigned kStepCount = 1000;
    const float kRegularization = 1000.0f;
    const float kForgettingFactor = 0.99f;

    Model model;
    model.Update();
    float error = model.mReferenceOutput - model.mOutput;
    float initialError = std::fabs( error / model.mOutput );

    ESN::AdaptiveFilterRLS filter( model.mInput.size(),
        kForgettingFactor, kRegularization );

    for ( int i = 0; i < kStepCount; ++ i )
    {
        model.Update();
        error = model.mReferenceOutput - model.mOutput;
        filter.Train(model.mW.data(), model.mOutput,
            model.mReferenceOutput, model.mInput.data());
    }

    EXPECT_LT( std::fabs( error / model.mOutput ), initialError );
}
