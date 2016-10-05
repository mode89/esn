#ifndef __ESN_CUDA_DEVICE_POINTER_H__
#define __ESN_CUDA_DEVICE_POINTER_H__

#include <memory>

namespace ESN {

    using DevicePointer = std::shared_ptr<float>;

    DevicePointer MakeDevicePointer(int byteSize);
    void MemcpyHostToDevice(
        const DevicePointer & dst, const float * src, int byteSize);
    void MemcpyDeviceToHost(
        float * dst, const DevicePointer & src, int bytesize);

} // namespace ESN

#endif // __ESN_CUDA_DEVICE_POINTER_H__
