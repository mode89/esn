#ifndef __ESN_CUDA_WRAP_H__
#define __ESN_CUDA_WRAP_H__

void wrap_sfillv(int n, const float * alpha, float * x);
void wrap_srcp(float *);
void wrap_stanhv(int n, float * v);
void wrap_srandv_helper(int n, const float * a, const float * b, float * x);

#endif // __ESN_CUDA_WRAP_H__
