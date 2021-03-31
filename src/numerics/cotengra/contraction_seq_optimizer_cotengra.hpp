#ifndef EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_COTENGRA_HPP_
#define EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_COTENGRA_HPP_

#include "contraction_seq_optimizer.hpp"
#include "Identifiable.hpp"

namespace exatn {
namespace numerics {

class ContractionSeqOptimizerCotengra : public ContractionSeqOptimizer,
                                        public Identifiable {
public:
  virtual double determineContractionSequence(
      const TensorNetwork &network, std::list<ContrTriple> &contr_seq,
      std::function<unsigned int()> intermediate_num_generator,
      uint64_t target_slice_size, std::list<SliceIndex> &slice_inds) override;

  virtual double determineContractionSequence(
      const TensorNetwork &network, std::list<ContrTriple> &contr_seq,
      std::function<unsigned int()> intermediate_num_generator) override {
    // Default is 2 GB
    constexpr uint64_t DEFAULT_SLICE_SIZE = 2 * (1ULL << 30);
    // Just ignore slice_inds result if using this method.
    std::list<SliceIndex> ignored_slice_inds;
    return determineContractionSequence(network, contr_seq,
                                        intermediate_num_generator,
                                        DEFAULT_SLICE_SIZE, ignored_slice_inds);
  }

  const std::string name() const override { return "cotengra"; }
  const std::string description() const override { return ""; }
  static std::unique_ptr<ContractionSeqOptimizer> createNew();
};
} // namespace numerics
} // namespace exatn

#endif // EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_COTENGRA_HPP_
