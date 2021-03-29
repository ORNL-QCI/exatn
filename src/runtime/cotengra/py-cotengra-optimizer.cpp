#include "contraction_seq_optimizer.hpp"
#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"

using namespace cppmicroservices;

namespace exatn {

namespace numerics {

class ContractionSeqOptimizerCotengra : public ContractionSeqOptimizer {
public:
  virtual double determineContractionSequence(
      const TensorNetwork &network, std::list<ContrTriple> &contr_seq,
      std::function<unsigned int()> intermediate_num_generator) override {
    return 0.0;
  }
};

} // namespace numerics
} // namespace exatn

namespace {

/**
 */
class US_ABI_LOCAL CotengraActivator : public BundleActivator {

public:
  CotengraActivator() {}

  /**
   */
  void Start(BundleContext context) {}

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(CotengraActivator)
