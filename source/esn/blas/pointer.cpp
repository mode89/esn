#include <cstring>
#include <esn/pointer.h>

namespace ESN {

    pointer make_pointer(std::size_t byteSize)
    {
        return std::shared_ptr<void>(
            std::malloc(byteSize),
            [] (void * p) { std::free(p); });
    }

    void memcpy(const pointer & dst, const void * src,
        std::size_t byteSize)
    {
        std::memcpy(dst.get(), src, byteSize);
    }

    void memcpy(void * dst, const const_pointer & src,
        std::size_t byteSize)
    {
        std::memcpy(dst, src.get(), byteSize);
    }

} // namespace ESN
