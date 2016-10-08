__global__ void kernel_srcp(float * v)
{
    v[0] = 1.0f / v[0];
}

__global__ void kernel_stanhv(int n, float * v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        v[i] = tanh(v[i]);
}

void wrap_srcp(float * v)
{
    kernel_srcp<<<1,1>>>(v);
}

void wrap_stanhv(int n, float * v)
{
    const int blockSize = 128;
    const int gridSize = (n + blockSize - 1) / blockSize;
    kernel_stanhv<<<gridSize, blockSize>>>(n, v);
}
