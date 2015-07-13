#include <esn/esn.h>

int main()
{
    ESN::NetworkParamsNSLI params;
    params.neuronCount = 100;
    params.inputCount = 1;
    params.outputCount = 1;
    auto network = ESN::CreateNetwork( params );
    return 0;
}
