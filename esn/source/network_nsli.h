#ifndef __ESN_NETWORK_NSLI_H__
#define __ESN_NETWORK_NSLI_H__

#include <Eigen/Sparse>
#include <esn/network.h>
#include <adaptive_filter_rls.h>

namespace ESN {

    struct NetworkParamsNSLI;

    /**
     * Implementation of a network based on non-spiking linear integrator
     * neurons.
     */
    class NetworkNSLI : public Network
    {
    public:
        void
        SetInputs( const std::vector< float > & );

        void
        Step( float step );

        void
        CaptureOutput( std::vector< float > & output );

        void
        Train(
            const std::vector< std::vector< float > > & inputs,
            const std::vector< std::vector< float > > & outputs );

        void
        TrainOnline( const std::vector< float > & output );

    public:
        NetworkNSLI( const NetworkParamsNSLI & );
        ~NetworkNSLI();

    private:
        NetworkParamsNSLI mParams;
        Eigen::VectorXf mIn;
        Eigen::MatrixXf mWIn;
        Eigen::VectorXf mX;
        Eigen::SparseMatrix< float > mW;
        Eigen::VectorXf mOut;
        Eigen::MatrixXf mWOut;
        Eigen::MatrixXf mWFB;
        AdaptiveFilterRLS mAdaptiveFilter;
    };

} // namespace ESN

#endif // __ESN_NETWORK_NSLI_H__
