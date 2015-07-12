#include <gtest/gtest.h>
#include <esn/create_network_nsli.h>
#include <esn/network.h>

TEST( ESN, CreateNetworkNSLI )
{
    ESN::NetworkParamsNSLI params;
    params.inputCount = 10;
    params.neuronCount = 100;
    params.outputCount = 10;
    std::unique_ptr< ESN::Network > network = CreateNetwork( params );
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
