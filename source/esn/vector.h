#ifndef __ESN_SOURCE_VECTOR_H__
#define __ESN_SOURCE_VECTOR_H__

#include <esn/pointer.h>

namespace ESN {

    template <class T>
    class vector
    {
    public:
        vector(std::size_t size)
            : m_size(size)
            , m_ptr(make_pointer<T>(size))
            , m_off(0)
            , m_inc(1)
        {}

        vector(
            const pointer<T> & ptr,
            std::size_t size,
            std::size_t off = 0,
            std::size_t inc = 1)
            : m_size(size)
            , m_ptr(ptr)
            , m_off(off)
            , m_inc(inc)
        {}

        std::size_t size() const { return m_size; }
        std::size_t inc() const { return m_inc; }
        const pointer<T> & ptr() { return m_ptr; }
        const const_pointer<T> & ptr() const { return m_ptr; }
        T * data() { return m_ptr.get() + m_off; }
        const T * data() const { return m_ptr.get() + m_off; }

    private:
        std::size_t m_size;
        const pointer<T> m_ptr;
        std::size_t m_off;
        std::size_t m_inc;
    };

} // namespace ESN

#endif // __ESN_SOURCE_VECTOR_H__
