#ifndef __ESN_POINTER_H__
#define __ESN_POINTER_H__

#include <memory>

namespace ESN {

    using pointer = std::shared_ptr<float>;

    pointer make_pointer(int byteSize);
    void memcpy(const pointer & dst, const float * src, int byteSize);
    void memcpy(float * dst, const pointer & src, int byteSize);

} // namespace ESN

#endif // __ESN_POINTER_H__
