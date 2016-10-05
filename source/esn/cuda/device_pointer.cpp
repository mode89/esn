#include <cuda_runtime.h>
#include <esn/cuda/debug.h>
#include <esn/cuda/device_pointer.h>

namespace ESN {

    DevicePointer MakeDevicePointer(int size)
    {
        float * pointer = nullptr;
        VCU(cudaMalloc, &pointer, size);
        return std::shared_ptr<float>(pointer, [] (float * p) {
            VCU(cudaFree, p);
        });
    }

} // namespace ESN
