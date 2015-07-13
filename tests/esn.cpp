#include <gtest/gtest.h>
#include <esn/create_network_nsli.h>
#include <esn/network.h>
#include <random>

TEST( ESN, CreateNetworkNSLI )
{
    ESN::NetworkParamsNSLI params;
    params.inputCount = 10;
    params.neuronCount = 100;
    params.outputCount = 10;
    std::unique_ptr< ESN::Network > network = CreateNetwork( params );
}

TEST( ESN, SetInputsNSLI )
{
    ESN::NetworkParamsNSLI params;
    params.inputCount = 25;
    params.neuronCount = 100;
    params.outputCount = 17;
    auto network = CreateNetwork( params );

    std::default_random_engine randomEngine;
    std::uniform_real_distribution<float > randomDist( -1.0f, 1.0f );
    std::vector< float > inputs( params.inputCount );
    for ( int i = 0; i < params.inputCount; ++ i )
        inputs[i] = randomDist( randomEngine );
}

TEST( ESN, StepNSLI )
{
    ESN::NetworkParamsNSLI params;
    params.inputCount = 11;
    params.neuronCount = 100;
    params.outputCount = 15;
    std::unique_ptr< ESN::Network > network = CreateNetwork( params );
    network->Step( 0.1f );
}

TEST( ESN, TrainNSLI )
{
    const unsigned kSampleCount = 100;

    ESN::NetworkParamsNSLI params;
    params.inputCount = 1;
    params.neuronCount = 100;
    params.outputCount = 1;
    auto network = CreateNetwork( params );

    std::vector< std::vector< float > > inputs( kSampleCount );
    std::vector< std::vector< float > > outputs( kSampleCount );
    for ( int i = 0; i < kSampleCount; ++ i )
    {
        inputs[i].resize( params.inputCount );
        inputs[i][0] = 1.0f;
        outputs[i].resize( params.outputCount );
        outputs[i][0] = 1.0f;
    }

    network->Train( inputs, outputs );
}
