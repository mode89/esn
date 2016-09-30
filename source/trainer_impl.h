#ifndef __ESN_SOURCE_TRAINER_IMPL_H__
#define __ESN_SOURCE_TRAINER_IMPL_H__

#include <esn/trainer.h>

namespace ESN {

    class AdaptiveFilterRLS;
    class NetworkNSLI;

    class TrainerImpl: public Trainer
    {
    public:
        TrainerImpl(
            const TrainerParams &,
            const std::shared_ptr<Network> &);

        void TrainSingleOutputOnline(
            unsigned index,
            float value,
            bool force) override;

        void TrainOnline(
            const std::vector<float> & output,
            bool forceOutput) override;

    private:
        TrainerParams mParams;
        std::shared_ptr<NetworkNSLI> mNetwork;
        std::vector<std::shared_ptr<AdaptiveFilterRLS>> mAdaptiveFilter;
    };

} // namespace ESN

#endif // __ESN_SOURCE_TRAINER_IMPL_H__
