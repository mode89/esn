#ifndef __ESN_EXCEPTIONS_H__
#define __ESN_EXCEPTIONS_H__

#include <stdexcept>

namespace ESN {

    class OutputIsNotFinite : public std::domain_error
    {
    public:
        OutputIsNotFinite()
            : std::domain_error("One or more outputs of the network are "
                "not finite values.")
        {}
    };

} // namespace ESN

#endif // __ESN_EXCEPTIONS_H__
