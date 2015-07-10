#include <gtest/gtest.h>
#include <esn/create_network_nsli.h>
#include <esn/network.h>

TEST( ESN, CreateNetworkNSLI )
{
    ESN::NetworkParamsNSLI params;
    std::unique_ptr< ESN::Network > network = CreateNetwork( params );
}
