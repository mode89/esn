#include <gtest/gtest.h>
#include <esn/network_nsli.h>
#include <esn/network.h>
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
    std::shared_ptr<ESN::Network> network = CreateNetwork(params);
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
    std::shared_ptr<ESN::Network> network = CreateNetwork(params);
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
        Randomize(outputs, -0.5f, 0.5f);
        network->TrainOnline(outputs, false);
    }
}

TEST(ESN, TransformOutput)
{
    ESN::NetworkParamsNSLI params;
    params.inputCount = 32;
    params.neuronCount = 64;
    params.outputCount = 16;
    auto network = CreateNetwork(params);

    std::vector<float> outputs_min(params.outputCount);
    Randomize(outputs_min, -10.0f, -1.0f);
    std::vector<float> outputs_max(params.outputCount);
    Randomize(outputs_max, 1.0f, 10.0f);

    std::vector<float> scale(params.outputCount);
    for (int i = 0; i < params.outputCount; ++i)
        scale[i] = outputs_max[i] - outputs_min[i];
    network->SetOutputScale(scale);

    std::vector<float> bias(params.outputCount);
    for (int i = 0; i < params.outputCount; ++i)
        bias[i] = (outputs_max[i] + outputs_min[i]) / 2.0f;
    network->SetOutputBias(bias);

    std::vector<float> inputs(params.inputCount);
    std::vector<float> outputs(params.outputCount);
    for (int s = 0; s < 100; ++ s)
    {
        Randomize(inputs, -1.0f, 1.0f);
        network->SetInputs(inputs);
        network->Step(1.0f);

        for (int i = 0; i < params.outputCount; ++i)
        {
            std::uniform_real_distribution<float> dist(
                outputs_min[i], outputs_max[i]);
            outputs[i] = dist(sRandomEngine);
        }

        network->TrainOnline(outputs, false);
    }
}
