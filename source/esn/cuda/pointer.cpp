#include <cuda_runtime.h>
#include <esn/cuda/debug.h>
#include <esn/pointer.h>

namespace ESN {

    pointer make_pointer(std::size_t size)
    {
        float * devicePointer = nullptr;
        VCU(cudaMalloc, &devicePointer, size);
        return std::shared_ptr<float>(devicePointer, [] (float * p) {
            VCU(cudaFree, p);
        });
    }

    pointer make_pointer(const float & value)
    {
        pointer retval = make_pointer(sizeof(float));
        memcpy(retval, value);
        return retval;
    }

    pointer make_pointer(const std::vector<float> & value)
    {
        pointer retval = make_pointer(value.size() * sizeof(float));
        memcpy(retval, value);
        return retval;
    }

    void memcpy(const pointer & dst, const float * src,
        std::size_t byteSize)
    {
        VCU(cudaMemcpy, dst.get(), src, byteSize, cudaMemcpyHostToDevice);
    }

    void memcpy(float * dst, const const_pointer & src,
        std::size_t byteSize)
    {
        VCU(cudaMemcpy, dst, src.get(), byteSize, cudaMemcpyDeviceToHost);
    }

    void memcpy(const pointer & dst, const float & src)
    {
        memcpy(dst, &src, sizeof(float));
    }

    void memcpy(float & dst, const const_pointer & src)
    {
        memcpy(&dst, src, sizeof(float));
    }

    void memcpy(const pointer & dst, const std::vector<float> & src)
    {
        memcpy(dst, src.data(), src.size() * sizeof(float));
    }

    void memcpy(std::vector<float> & dst, const const_pointer & src)
    {
        memcpy(dst.data(), src, dst.size() * sizeof(float));
    }

} // namespace ESN
