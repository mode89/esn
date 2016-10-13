#ifndef __ESN_SOURCE_SCALAR_H__
#define __ESN_SOURCE_SCALAR_H__

#include <esn/pointer.h>
#include <esn/vector.h>

namespace ESN {

    template <class T>
    class scalar: public vector<T>
    {
    public:
        scalar(T value)
            : vector<T>(make_pointer<T>(value), 1)
        {}
    };

} // namespace ESN

#endif // __ESN_SOURCE_SCALAR_H__
