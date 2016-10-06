#ifndef __ESN_POINTER_H__
#define __ESN_POINTER_H__

#include <memory>
#include <vector>

namespace ESN {

    using pointer = std::shared_ptr<float>;
    using const_pointer = std::shared_ptr<const float>;

    pointer make_pointer(std::size_t byteSize);
    pointer make_pointer(const float &);
    pointer make_pointer(const std::vector<float> &);

    // General copy
    void memcpy(const pointer & dst, const float * src,
        std::size_t byteSize);
    void memcpy(float * dst, const const_pointer & src,
        std::size_t byteSize);

    // Copy between single float and pointer
    void memcpy(const pointer & dst, const float & src);
    void memcpy(float & dst, const const_pointer & src);

    // Copy between std::vector and pointer
    void memcpy(const pointer & dst, const std::vector<float> & src);
    void memcpy(std::vector<float> & dst, const const_pointer & src);

} // namespace ESN

#endif // __ESN_POINTER_H__
