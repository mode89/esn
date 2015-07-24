#ifndef __ESN_NETWORK_H__
#define __ESN_NETWORK_H__

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void
esnNetworkSetInputs( void * network,
    float * inputs, int inputCount );

void
esnNetworkStep( void * network,
    float step );

void
esnNetworkCaptureOutput( void * network,
    float * outputs, int outputCount );

void
esnNetworkTrainOnline( void * network,
    float * outputs, int outputCount, bool forceOutpus );

void
esnNetworkDestruct( void * network );

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __ESN_NETWORK_H__
