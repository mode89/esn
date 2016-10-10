__global__ void kernel_sfillv(int n, const float * alpha, float * x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = *alpha;
}

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

__global__ void kernel_srandv_helper(
    int n, const float * a, const float * b, float * x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = x[i] * (*b - *a) + *a;
}

__global__ void kernel_srandspv_helper(
    int n, const float * sparsity, const float * spx, float * x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        if (spx[i] < *sparsity)
            x[i] = 0.0f;
}

void wrap_sfillv(int n, const float * alpha, float * x)
{
    const int blockSize = 128;
    const int gridSize = (n + blockSize - 1) / blockSize;
    kernel_sfillv<<<gridSize, blockSize>>>(n, alpha, x);
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

void wrap_srandv_helper(int n, const float * a, const float * b, float * x)
{
    const int blockSize = 128;
    const int gridSize = (n + blockSize - 1) / blockSize;
    kernel_srandv_helper<<<gridSize, blockSize>>>(n, a, b, x);
}

void wrap_srandspv_helper(int n, const float * sparsity,
    const float * spx, float * x)
{
    const int blockSize = 128;
    const int gridSize = (n + blockSize - 1) / blockSize;
    kernel_srandspv_helper<<<gridSize, blockSize>>>(n, sparsity, spx, x);
}
