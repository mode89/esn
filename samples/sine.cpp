#include <esn/esn.h>

static const unsigned kNeuronCount = 100;

int main()
{
    ESN::NetworkParamsNSLI params;
    params.inputCount = 1;
    params.neuronCount = kNeuronCount;
    params.outputCount = 1;
    std::unique_ptr< ESN::Network > network = ESN::CreateNetwork( params );

    return 0;
}
