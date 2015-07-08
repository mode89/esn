#ifndef __ESN_NETWORK_H__
#define __ESN_NETWORK_H__

namespace ESN {

    class Network
    {
    public:
        virtual void Step( float step ) = 0;

        virtual ~Network() {}
    };

} // namespace ESN

#endif // __ESN_NETWORK_H__
