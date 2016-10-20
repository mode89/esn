#ifndef __ESN_CUDA_WRAP_H__
#define __ESN_CUDA_WRAP_H__

void wrap_sfillv(int n, const float * alpha, float * x);
void wrap_srcp(float *);
void wrap_stanhv(int n, float * v);
void wrap_srandv_helper(int n, const float * a, const float * b, float * x);
void wrap_srandspv_helper(int n, const float * sparsity,
    const float * spx, float * x);
void wrap_sprodvv(int n, const float * x, float * y);
void wrap_sdivvv(int n, float * x, const float * y);

#endif // __ESN_CUDA_WRAP_H__
