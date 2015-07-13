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
