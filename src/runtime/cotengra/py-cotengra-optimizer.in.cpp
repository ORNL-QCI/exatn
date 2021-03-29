#include "contraction_seq_optimizer.hpp"
#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include <dlfcn.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "tensor_network.hpp"

using namespace cppmicroservices;
namespace py = pybind11;

namespace exatn {

namespace numerics {

class ContractionSeqOptimizerCotengra : public ContractionSeqOptimizer, public Identifiable {
public:
  virtual double determineContractionSequence(
      const TensorNetwork &network, std::list<ContrTriple> &contr_seq,
      std::function<unsigned int()> intermediate_num_generator) override {
    if (!initialized) {
      guard = std::make_shared<py::scoped_interpreter>();
      libpython_handle = dlopen("@PYTHON_LIB_NAME@", RTLD_LAZY | RTLD_GLOBAL);
      initialized = true;
    }

    for(auto it = network.cbegin(); it != network.cend(); ++it) {
      const auto tensorId = it->first;
      std::cout << "Tensor ID: " << tensorId << "\n";
    }
    


    return 0.0;
  }

  virtual const std::string name() const override { return "cotengra"; }
  virtual const std::string description() const override { return ""; }

private:
  bool initialized = false;
  std::shared_ptr<py::scoped_interpreter> guard;
  void *libpython_handle = nullptr;
};

} // namespace numerics
} // namespace exatn

namespace {

/**
 */
class US_ABI_LOCAL CotengraActivator : public BundleActivator {

public:
  CotengraActivator() {}

  void Start(BundleContext context) {
    context.RegisterService<exatn::numerics::ContractionSeqOptimizer>(
        std::make_shared<exatn::numerics::ContractionSeqOptimizerCotengra>());
  }

  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(CotengraActivator)
