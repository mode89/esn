#include <gtest/gtest.h>
#include <esn/network_nsli.hpp>
#include <esn/network.hpp>
#include <random>

std::default_random_engine sRandomEngine;

static void Randomize(std::vector<float> & v, float min, float max)
{
    std::uniform_real_distribution<float> dist(min, max);
    for (auto & val : v)
        val = dist(sRandomEngine);
}

TEST( ESN, CreateNetworkNSLI )
{
    ESN::NetworkParamsNSLI params;
    params.inputCount = 10;
    params.neuronCount = 100;
    params.outputCount = 10;
    std::unique_ptr< ESN::Network > network = CreateNetwork( params );
}

TEST(ESN, SetInputs)
{
    ESN::NetworkParamsNSLI params;
    params.inputCount = 25;
    params.neuronCount = 100;
    params.outputCount = 17;
    auto network = CreateNetwork( params );

    std::vector< float > inputs( params.inputCount );
    for (int s = 0; s < 100; ++ s)
    {
        Randomize(inputs, -1.0f, 1.0f);
        network->SetInputs(inputs);
    }
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

TEST(ESN, TrainOnline)
{
    ESN::NetworkParamsNSLI params;
    params.inputCount = 32;
    params.neuronCount = 64;
    params.outputCount = 16;
    auto network = CreateNetwork(params);

    std::vector<float> inputs(params.inputCount);
    std::vector<float> outputs(params.outputCount);
    for (int s = 0; s < 100; ++ s)
    {
        Randomize(inputs, -1.0f, 1.0f);
        network->SetInputs(inputs);
        network->Step(1.0f);
        Randomize(outputs, -0.7f, 0.7f);
        network->TrainOnline(outputs, false);
    }
}

TEST(ESN, NoFeedback)
{
    ESN::NetworkParamsNSLI params;
    params.inputCount = 32;
    params.neuronCount = 64;
    params.outputCount = 16;
    params.hasOutputFeedback = false;
    auto network = CreateNetwork(params);

    std::vector<float> inputs(params.inputCount);
    std::vector<float> outputs(params.outputCount);
    for (int s = 0; s < 100; ++ s)
    {
        Randomize(inputs, -1.0f, 1.0f);
        network->SetInputs(inputs);
        network->Step(1.0f);
        Randomize(outputs, -0.7f, 0.7f);
        network->TrainOnline(outputs, false);
    }
}
