#ifndef __ESN_POINTER_H__
#define __ESN_POINTER_H__

#include <memory>
#include <vector>

namespace ESN {

    template <class T>
    using pointer = std::shared_ptr<T>;
    template <class T>
    using const_pointer = std::shared_ptr<const T>;

    template <class T>
    pointer<T> make_pointer(std::size_t elemCount);
    template <class T>
    pointer<T> make_pointer(const T &);
    template <class T>
    pointer<T> make_pointer(const std::vector<T> &);

    // General copy
    template <class T>
    void memcpy(const pointer<T> & dst, const T * src,
        std::size_t elemCount);
    template <class T>
    void memcpy(T * dst, const const_pointer<T> & src,
        std::size_t elemCount);

    // Copy between single float and pointer
    template <class T>
    void memcpy(const pointer<T> & dst, const T & src);
    template <class T>
    void memcpy(T & dst, const const_pointer<T> & src);

    // Copy between std::vector and pointer
    template <class T>
    void memcpy(const pointer<T> & dst, const std::vector<T> & src);
    template <class T>
    void memcpy(std::vector<T> & dst, const const_pointer<T> & src);

} // namespace ESN

#endif // __ESN_POINTER_H__
