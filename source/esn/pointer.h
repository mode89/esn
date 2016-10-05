#ifndef __ESN_POINTER_H__
#define __ESN_POINTER_H__

#include <memory>

namespace ESN {

    using pointer = std::shared_ptr<float>;
    using const_pointer = std::shared_ptr<const float>;

    pointer make_pointer(int byteSize);
    void memcpy(const pointer & dst, const float * src, int byteSize);
    void memcpy(float * dst, const const_pointer & src, int byteSize);

} // namespace ESN

#endif // __ESN_POINTER_H__
