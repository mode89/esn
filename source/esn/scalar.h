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

        scalar<T> & operator=(const T & value)
        {
            if (vector<T>::off() != 0)
                std::runtime_error(
                    "Cannot copy to scalar with non-zero offset");
            memcpy(vector<T>::ptr(), &value, 1);
            return *this;
        }
    };

} // namespace ESN

#endif // __ESN_SOURCE_SCALAR_H__
