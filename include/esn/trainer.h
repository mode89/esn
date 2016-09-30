#ifndef __ESN_TRAINER_H__
#define __ESN_TRAINER_H__

#include <esn/export.h>
#include <memory>
#include <vector>

namespace ESN {

    class Network;

    struct TrainerParams
    {
        float forgettingFactor;
        float initialCovariance;

        TrainerParams()
            : forgettingFactor(1.0f)
            , initialCovariance(1000.0f)
        {}
    };

    class Trainer
    {
    public:
        virtual ESN_EXPORT void
        TrainSingleOutputOnline(
            unsigned index,
            float value,
            bool force) = 0;

        virtual ESN_EXPORT void
        TrainOnline(
            const std::vector<float> & output,
            bool forceOutput) = 0;

        virtual ~Trainer() {}
    };

    std::shared_ptr<Trainer> CreateTrainer(
        const TrainerParams &,
        const std::shared_ptr<Network> &);

} // namespace ESN

#endif // __ESN_TRAINER_H__
