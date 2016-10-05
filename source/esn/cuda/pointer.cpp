#include <cuda_runtime.h>
#include <esn/cuda/debug.h>
#include <esn/pointer.h>

namespace ESN {

    pointer make_pointer(int size)
    {
        float * devicePointer = nullptr;
        VCU(cudaMalloc, &devicePointer, size);
        return std::shared_ptr<float>(devicePointer, [] (float * p) {
            VCU(cudaFree, p);
        });
    }

    void memcpy(const pointer & dst, const float * src, int byteSize)
    {
        VCU(cudaMemcpy, dst.get(), src, byteSize, cudaMemcpyHostToDevice);
    }

    void memcpy(float * dst, const const_pointer & src, int byteSize)
    {
        VCU(cudaMemcpy, dst, src.get(), byteSize, cudaMemcpyDeviceToHost);
    }

} // namespace ESN
