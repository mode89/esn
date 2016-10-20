#ifndef __ESN_SOURCE_SCALAR_H__
#define __ESN_SOURCE_SCALAR_H__

#include <esn/pointer.h>

namespace ESN {

    template <class T>
    class vector;

    template <class T>
    class scalar: public vector<T>
    {
    public:
        scalar(T value)
            : vector<T>(1)
        {
            memcpy(vector<T>::ptr(), &value, sizeof(T));
        }

        scalar(const pointer & ptr, std::size_t off)
            : vector<T>(ptr, 1, off)
        {
        }

        scalar<T> & operator=(const T & value)
        {
            if (vector<T>::off() != 0)
                std::runtime_error(
                    "Cannot copy to scalar with non-zero offset");
            memcpy(vector<T>::ptr(), &value, sizeof(T));
            return *this;
        }

        explicit operator T() const
        {
            T retval;
            memcpy(&retval, vector<T>::ptr(), sizeof(T));
            return retval;
        }
    };

} // namespace ESN

#endif // __ESN_SOURCE_SCALAR_H__
