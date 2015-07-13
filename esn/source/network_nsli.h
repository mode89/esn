#ifndef __ESN_NETWORK_NSLI_H__
#define __ESN_NETWORK_NSLI_H__

#include <Eigen/Sparse>
#include <esn/network.h>

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
        Train(
            const std::vector< std::vector< float > > & inputs,
            const std::vector< std::vector< float > > & outputs );

    public:
        NetworkNSLI( const NetworkParamsNSLI & );
        ~NetworkNSLI();

    private:
        NetworkParamsNSLI mParams;
        Eigen::VectorXf mIn;
        Eigen::SparseMatrix< float > mWIn;
        Eigen::VectorXf mX;
        Eigen::SparseMatrix< float > mW;
        Eigen::VectorXf mOut;
        Eigen::SparseMatrix< float > mWOut;
    };

} // namespace ESN

#endif // __ESN_NETWORK_NSLI_H__
