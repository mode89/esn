#ifndef __ESN_CUDA_DEVICE_POINTER_H__
#define __ESN_CUDA_DEVICE_POINTER_H__

#include <memory>

namespace ESN {

    using DevicePointer = std::shared_ptr<float>;

    DevicePointer MakeDevicePointer(int size);

} // namespace ESN

#endif // __ESN_CUDA_DEVICE_POINTER_H__
