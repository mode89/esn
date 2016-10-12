#ifndef __ESN_SOURCE_SCALAR_H__
#define __ESN_SOURCE_SCALAR_H__

#include <esn/pointer.h>

namespace ESN {

    template <class T>
    class scalar
    {
    public:
        scalar(T value)
            : m_ptr(make_pointer<T>(value))
            , m_off(0)
        {}

        scalar(
            const pointer<T> & ptr,
            std::size_t off = 0)
            : m_ptr(ptr)
            , m_off(off)
        {}

        const pointer<T> & ptr() { return m_ptr; }
        const const_pointer<T> & ptr() const { return m_ptr; }
        T * data() { return m_ptr.get() + m_off; }
        const T * data() const { return m_ptr.get() + m_off; }

    private:
        const pointer<T> m_ptr;
        std::size_t m_off;
    };

} // namespace ESN

#endif // __ESN_SOURCE_SCALAR_H__
