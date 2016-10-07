__global__ void kernel_srcp(float * v)
{
    v[0] = 1.0f / v[0];
}

void wrap_srcp(float * v)
{
    kernel_srcp<<<1,1>>>(v);
}
