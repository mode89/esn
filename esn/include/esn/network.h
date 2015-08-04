#ifndef __ESN_NETWORK_H__
#define __ESN_NETWORK_H__

#include <esn/export.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

ESN_EXPORT void
esnNetworkSetInputs( void * network,
    float * inputs, int inputCount );

ESN_EXPORT void
esnNetworkSetInputScalings( void * network,
    float * scalings, int count );

ESN_EXPORT void
esnNetworkSetFeedbackScalings( void * network,
    float * scalings, int cound );

ESN_EXPORT void
esnNetworkStep( void * network,
    float step );

ESN_EXPORT void
esnNetworkCaptureActivations( void * network,
    float * activations, int neuronCount );

ESN_EXPORT void
esnNetworkCaptureOutput( void * network,
    float * outputs, int outputCount );

ESN_EXPORT void
esnNetworkTrainOnline( void * network,
    float * outputs, int outputCount, bool forceOutpus );

ESN_EXPORT void
esnNetworkDestruct( void * network );

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __ESN_NETWORK_H__
