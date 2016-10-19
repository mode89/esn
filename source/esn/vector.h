#ifndef __ESN_SOURCE_VECTOR_H__
#define __ESN_SOURCE_VECTOR_H__

#include <esn/pointer.h>
#include <vector>

namespace ESN {

    template <class T>
    class vector
    {
    public:
        vector(std::size_t size)
            : m_size(size)
            , m_ptr(make_pointer(size * sizeof(T)))
            , m_off(0)
            , m_inc(1)
        {}

        vector(
            const pointer & ptr,
            std::size_t size,
            std::size_t off = 0,
            std::size_t inc = 1)
            : m_size(size)
            , m_ptr(ptr)
            , m_off(off)
            , m_inc(inc)
        {}

        vector(const T * v, std::size_t size)
            : m_size(size)
            , m_inc(1)
            , m_ptr(make_pointer(size * sizeof(T)))
            , m_off(0)
        {
            memcpy(m_ptr, v, size * sizeof(T));
        }

        vector(const std::vector<T> & v)
            : vector(v.data(), v.size())
        {}

        vector<T> & operator=(const std::vector<T> & v)
        {
            if (v.size() != m_size)
                throw std::runtime_error("Cannot copy std::vector to "
                    "vector of different size");
            if (m_inc != 1)
                throw std::runtime_error("Cannot copy std::vector to "
                    "vector with non-unit increment");
            if (m_off != 0)
                throw std::runtime_error("Cannot copy std::vector to "
                    "vector with non-zero offset");
            memcpy(m_ptr, v.data(), m_size * sizeof(T));
            return *this;
        }

        vector<T> & operator=(vector<T> && other) = default;

        operator std::vector<T>() const
        {
            if (m_inc != 1)
                throw std::runtime_error("Cannot convert vector with "
                    "non-unit increment to std::vector");
            if (m_off != 0)
                throw std::runtime_error("Cannot convert vector with "
                    "non-zero offset to std::vector");
            std::vector<T> retval(m_size);
            memcpy(retval.data(), m_ptr, m_size * sizeof(T));
            return retval;
        }

        std::size_t size() const { return m_size; }
        std::size_t inc() const { return m_inc; }
        std::size_t off() const { return m_off; }
        const pointer & ptr() { return m_ptr; }
        const_pointer ptr() const { return m_ptr; }

    protected:
        std::size_t m_size;
        std::size_t m_inc;
        pointer m_ptr;
        std::size_t m_off;
    };

} // namespace ESN

#endif // __ESN_SOURCE_VECTOR_H__
