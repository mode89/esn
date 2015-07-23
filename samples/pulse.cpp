#include <esn/esn.h>
#include <random>

static const unsigned kNeuronCount = 100;
static const float kStep = 0.01f;
static const float kIntervalMin = 0.1f;
static const float kIntervalMax = 1.0f;

int main()
{
    ESN::NetworkParamsNSLI params;
    params.neuronCount = kNeuronCount;
    params.inputCount = 1;
    params.outputCount = 1;
    auto network = ESN::CreateNetwork( params );

    bool inputState = false;
    std::vector< float > input( 1 );
    float nextInterval = 0.0f;
    std::default_random_engine randomEngine;
    std::uniform_real_distribution< float > random(
        kIntervalMin, kIntervalMax );
    for ( float time = 0.0f; true; time += kStep )
    {
        if ( time > nextInterval )
        {
            inputState = !inputState;
            input[0] = inputState ? 1.0f : 0.0f;
            nextInterval += random( randomEngine );
        }
    }

    return 0;
}
