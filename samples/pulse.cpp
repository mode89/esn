#include <cmath>
#include <esn/esn.h>
#include <iomanip>
#include <iostream>
#include <random>

static const unsigned kNeuronCount = 100;
static const float kStep = 0.01f;
static const float kIntervalWidthMin = 0.1f;
static const float kIntervalWidthMax = 1.0f;
static const float kTargetIntervalWidth = 0.7f;
static const float kTargetIntervalError = 0.1f;
static const float kOutputPulseWidth = 0.1f;
static const float kOutputPulseMax = 1.0f;
static const float kOutputPulseThreshold = 0.0001f;

inline float Pulse( float x, float width, float max )
{
    float retval = max * std::exp( -std::pow(
        6.0f * ( x - width / 2.0f ) / width, 2.0f ) );
    return ( retval < kOutputPulseThreshold ) ? 0.0f : retval;
}

int main()
{
    ESN::NetworkParamsNSLI params;
    params.neuronCount = kNeuronCount;
    params.inputCount = 1;
    params.outputCount = 1;
    auto network = ESN::CreateNetwork( params );

    bool inputState = false;
    std::vector< float > input( 1 );
    float currentIntervalWidth = 0.0f;
    float nextInterval = 0.0f;
    std::default_random_engine randomEngine;
    std::uniform_real_distribution< float > random(
        kIntervalWidthMin, kIntervalWidthMax );

    std::vector< float > referenceOutput( 1 );
    float outputPulseStart = 0.0f;
    float outputPulseAmplitude = 0.0f;

    for ( float time = 0.0f; true; time += kStep )
    {
        if ( time > nextInterval )
        {
            if ( inputState &&
                 std::fabs( currentIntervalWidth - kTargetIntervalWidth ) <
                 kTargetIntervalError )
            {
                outputPulseStart = time;
                outputPulseAmplitude = ( kTargetIntervalError - std::fabs(
                    currentIntervalWidth - kTargetIntervalWidth ) ) /
                    kTargetIntervalError;
            }

            inputState = !inputState;
            input[0] = inputState ? 1.0f : 0.0f;

            currentIntervalWidth = random( randomEngine );
            nextInterval += currentIntervalWidth;
        }

        referenceOutput[0] = Pulse( time - outputPulseStart,
            kOutputPulseWidth, outputPulseAmplitude );

        std::cout <<
            std::setw( 3 ) << inputState <<
            std::endl;
    }

    return 0;
}
