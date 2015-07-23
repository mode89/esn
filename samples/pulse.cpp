#include <esn/esn.h>

static const unsigned kNeuronCount = 100;

int main()
{
    ESN::NetworkParamsNSLI params;
    params.neuronCount = kNeuronCount;
    params.inputCount = 1;
    params.outputCount = 1;
    auto network = ESN::CreateNetwork( params );
    return 0;
}
