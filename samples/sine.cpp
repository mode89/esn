#define _USE_MATH_DEFINES
#include <esn/esn.h>
#include <math.h>

static const unsigned kNeuronCount = 100;
static const float kSineFrequency = 1.0f;
static const float kSineStep = 0.0001f;
static const float kErrorThreshold = 0.001f;

int main()
{
    ESN::NetworkParamsNSLI params;
    params.inputCount = 1;
    params.neuronCount = kNeuronCount;
    params.outputCount = 1;
    std::unique_ptr< ESN::Network > network = ESN::CreateNetwork( params );

    std::vector< float > output( 1 );
    std::vector< float > actualOutput( 1 );
    for ( float time = 0.0f; true; time += kSineStep )
    {
        output[0] = sin( 2 * static_cast< float >( M_PI ) *
            kSineFrequency * time ) + 2.0f;

        network->Step( 0.1f );
        network->CaptureOutput( actualOutput );
        float error = fabs( ( actualOutput[0] - output[0] ) / output[0] );

        if ( error > kErrorThreshold )
        {
            network->TrainOnline( output );
        }
    }

    return 0;
}
