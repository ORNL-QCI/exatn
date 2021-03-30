#ifndef EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_COTENGRA_HPP_
#define EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_COTENGRA_HPP_

#include "contraction_seq_optimizer.hpp"
#include "Identifiable.hpp"

namespace pybind11 {
class scoped_interpreter;
}

namespace exatn {
namespace numerics {

class ContractionSeqOptimizerCotengra : public ContractionSeqOptimizer,
                                        public Identifiable {
public:
  virtual double determineContractionSequence(
      const TensorNetwork &network, std::list<ContrTriple> &contr_seq,
      std::function<unsigned int()> intermediate_num_generator) override;

  const std::string name() const override { return "cotengra"; }
  const std::string description() const override { return ""; }
  static std::unique_ptr<ContractionSeqOptimizer> createNew();

private:
  bool initialized = false;
  std::shared_ptr<pybind11::scoped_interpreter> guard;
  void *libpython_handle = nullptr;
};
} // namespace numerics
} // namespace exatn

#endif // EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_COTENGRA_HPP_
