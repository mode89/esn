#ifndef __ESN_SOURCE_ESN_MATH_H__
#define __ESN_SOURCE_ESN_MATH_H__

extern "C" {
    #include <cblas.h>
}

#include <lapacke.h>
#include <random>

namespace ESN {

    extern std::default_random_engine sRandomEngine;

    void RandomUniform(float * v, int size, float a, float b);
    void Constant(float * v, int size, float value);
    void TanhEwise(float * v, int size);
    void ProductEwise(float * out, const float * in, int size);
    void SumEwise(float * out, const float * a, const float * b, int size);

} // namespace ESN

#endif // __ESN_SOURCE_ESN_MATH_H__
