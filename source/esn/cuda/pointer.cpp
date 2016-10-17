#include <cuda_runtime.h>
#include <esn/cuda/debug.h>
#include <esn/pointer.h>

namespace ESN {

    pointer make_pointer(std::size_t byteSize)
    {
        void * devicePointer = nullptr;
        VCU(cudaMalloc, &devicePointer, byteSize);
        return std::shared_ptr<float>(
            static_cast<float*>(devicePointer),
            [] (void * p) { VCU(cudaFree, p); });
    }

    void memcpy(const pointer & dst, const void * src,
        std::size_t byteSize)
    {
        VCU(cudaMemcpy, dst.get(), src, byteSize, cudaMemcpyHostToDevice);
    }

    void memcpy(void * dst, const const_pointer & src,
        std::size_t byteSize)
    {
        VCU(cudaMemcpy, dst, src.get(), byteSize, cudaMemcpyDeviceToHost);
    }

} // namespace ESN
