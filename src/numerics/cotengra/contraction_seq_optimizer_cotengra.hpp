#ifndef EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_COTENGRA_HPP_
#define EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_COTENGRA_HPP_

#include "contraction_seq_optimizer.hpp"
#include "Identifiable.hpp"

namespace exatn {
namespace numerics {

class ContractionSeqOptimizerCotengra : public ContractionSeqOptimizer,
                                        public Identifiable {
public:

  static constexpr uint64_t DEFAULT_SLICE_SIZE = 2 * (1ULL << 30); //bytes

  virtual double determineContractionSequence(
      const TensorNetwork & network,
      std::list<ContrTriple> & contr_seq,
      std::function<unsigned int()> intermediate_num_generator,
      uint64_t target_slice_size,
      std::list<SliceIndex> & slice_inds) override;

  virtual double determineContractionSequence(
      const TensorNetwork & network,
      std::list<ContrTriple> & contr_seq,
      std::function<unsigned int()> intermediate_num_generator) override {
    // Just ignore slice_inds result if using this method
    std::list<SliceIndex> ignored_slice_inds;
    double flops = determineContractionSequence(network, contr_seq, intermediate_num_generator,
                                                DEFAULT_SLICE_SIZE, ignored_slice_inds);
    /* DEBUG: Print the contraction path and sliced indices:
    std::cout << "#DEBUG(ContractionSeqOptimizerCotengra): Contraction path:" << std::endl;
    for(const auto & triple: contr_seq)
     std::cout << triple.result_id << " " << triple.left_id << " " << triple.right_id << std::endl;
    std::cout << "#DEBUG(ContractionSeqOptimizerCotengra): Sliced indices:" << std::endl;
    for(const auto & slice_index: ignored_slice_inds)
     std::cout << slice_index.tensor_id << " " << slice_index.leg_id << std::endl;
    std::cout << "#DEBUG(ContractionSeqOptimizerCotengra): Flops = " << flops << std::endl;
    */
    return flops;
  }

  const std::string name() const override { return "cotengra"; }
  const std::string description() const override { return ""; }
  static std::unique_ptr<ContractionSeqOptimizer> createNew();
};
} // namespace numerics
} // namespace exatn

#endif // EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_COTENGRA_HPP_
