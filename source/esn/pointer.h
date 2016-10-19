#ifndef __ESN_POINTER_H__
#define __ESN_POINTER_H__

#include <memory>
#include <vector>

namespace ESN {

    // TODO implement as void
    using pointer = std::shared_ptr<float>;
    using const_pointer = std::shared_ptr<const float>;

    pointer make_pointer(std::size_t byteSize);

    void memcpy(
        const pointer & dst,
        const void * src,
        std::size_t byteSize);

    void memcpy(
        void * dst,
        const const_pointer & src,
        std::size_t byteSize);

} // namespace ESN

#endif // __ESN_POINTER_H__
