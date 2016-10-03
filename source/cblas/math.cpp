#include <esn/math.h>
#include <random>

namespace ESN {

    std::default_random_engine sRandomEngine;

    void RandomUniform(float * v, int size, float a, float b)
    {
        std::uniform_real_distribution<float> dist(a, b);
        for (int i = 0; i < size; ++ i)
            v[i] = dist(sRandomEngine);
    }

    void Constant(float * v, int size, float value)
    {
        for (int i = 0; i < size; ++ i)
            v[i] = value;
    }

    void TanhEwise(float * v, int size)
    {
        for (int i = 0; i < size; ++ i)
            v[i] = std::tanh(v[i]);
    }

    void ProductEwise(float * out, const float * in, int size)
    {
        for (int i = 0; i < size; ++ i)
            out[i] *= in[i];
    }

    void SumEwise(float * out, const float * a, const float * b, int size)
    {
        for (int i = 0; i < size; ++ i)
            out[i] = a[i] + b[i];
    }

} // namespace ESN
