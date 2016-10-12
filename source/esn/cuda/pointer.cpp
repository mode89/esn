#include <cuda_runtime.h>
#include <esn/cuda/debug.h>
#include <esn/pointer.h>

namespace ESN {

    template <class T>
    pointer<T> make_pointer(std::size_t elemCount)
    {
        T * devicePointer = nullptr;
        VCU(cudaMalloc, &devicePointer, elemCount * sizeof(T));
        return std::shared_ptr<T>(devicePointer, [] (T * p) {
            VCU(cudaFree, p);
        });
    }

    template pointer<float> make_pointer(std::size_t);

    template <class T>
    pointer<T> make_pointer(const T & value)
    {
        pointer<T> retval = make_pointer<float>(1);
        memcpy(retval, value);
        return retval;
    }

    template pointer<float> make_pointer(const float &);

    template <class T>
    pointer<T> make_pointer(const std::vector<T> & value)
    {
        pointer<T> retval = make_pointer<float>(value.size());
        memcpy(retval, value);
        return retval;
    }

    template pointer<float> make_pointer(const std::vector<float> &);

    template <class T>
    void memcpy(const pointer<T> & dst, const T * src,
        std::size_t elemCount)
    {
        VCU(cudaMemcpy, dst.get(), src, elemCount * sizeof(T),
            cudaMemcpyHostToDevice);
    }

    template void memcpy(
        const pointer<float> &, const float *, std::size_t);

    template <class T>
    void memcpy(T * dst, const const_pointer<T> & src,
        std::size_t elemCount)
    {
        VCU(cudaMemcpy, dst, src.get(), elemCount * sizeof(T),
            cudaMemcpyDeviceToHost);
    }

    template void memcpy(
        float *, const const_pointer<float> &, std::size_t);

    template <class T>
    void memcpy(const pointer<T> & dst, const T & src)
    {
        memcpy(dst, &src, 1);
    }

    template void memcpy(const pointer<float> &, const float &);

    template <class T>
    void memcpy(T & dst, const const_pointer<T> & src)
    {
        memcpy(&dst, src, 1);
    }

    template void memcpy(float &, const const_pointer<float> &);

    template <class T>
    void memcpy(const pointer<T> & dst, const std::vector<T> & src)
    {
        memcpy(dst, src.data(), src.size());
    }

    template void memcpy(
        const pointer<float> &, const std::vector<float> &);

    template <class T>
    void memcpy(std::vector<T> & dst, const const_pointer<T> & src)
    {
        memcpy(dst.data(), src, dst.size());
    }

    template void memcpy(
        std::vector<float> &, const const_pointer<float> &);

} // namespace ESN
