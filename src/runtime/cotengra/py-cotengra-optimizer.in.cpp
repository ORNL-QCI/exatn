#include "contraction_seq_optimizer.hpp"
#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include <dlfcn.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "tensor_network.hpp"

using namespace cppmicroservices;
using namespace pybind11::literals;
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

    auto nx = py::module::import("networkx");
    auto oe = py::module::import("opt_einsum");
    auto ctg = py::module::import("cotengra");
    
    auto graph = nx.attr("Graph")();
    auto tensor_rank_map = py::dict();
 
    for (auto it = network.cbegin(); it != network.cend(); ++it) {
      const auto tensorId = it->first;
      auto tensor_conn = it->second;
      // std::cout << "Tensor ID: " << tensorId << "\n";
      graph.attr("add_node")(tensorId);
      auto legs = tensor_conn.getTensorLegs();
      tensor_rank_map[py::int_(tensorId)] = tensor_conn.getRank();
      if (tensorId > 0) {
        for (const auto &leg : legs) {
          graph.attr("add_edge")(tensorId, leg.getTensorId());
        }
      }
    }
    // py::print(tensor_rank_map);
    
    auto globals = py::globals();
    globals["network_gragh"] = graph;
    globals["tensor_rank_data"] = tensor_rank_map;
    globals["oe"] = oe;
    {
      auto py_src = R"#(
graph = globals()['network_gragh']  
edge2ind_map = {tuple(sorted(e)): oe.get_symbol(i) for i, e in enumerate(graph.edges)}
)#";

      auto locals = py::dict();
      py::exec(py_src, py::globals(), locals);
      // py::print(locals["edge2ind_map"]);
      globals["edge2ind"] = locals["edge2ind_map"];
    }
  auto py_src = R"#(
graph = globals()['network_gragh']
rank_data = globals()['tensor_rank_data']
shape_map = {}
for tensor_id in rank_data:
  shape_map[tensor_id] = ((2,) * rank_data[tensor_id])

from opt_einsum.contract import Shaped
inputs = []
output = {}
node_list = []
for nd in graph.nodes:
  if nd == 0:
    output = {edge2ind[tuple(sorted(e))] for e in graph.edges(nd)}
  else:
    inputs.append({edge2ind[tuple(sorted(e))] for e in graph.edges(nd)})
    node_list.append(nd)

eq = (",".join(["".join(i) for i in inputs]) + "->{}".format("".join(output)))
shapes = []
for nd in graph.nodes:
  if nd != 0:
    shapes.append(shape_map[nd])    
views = list(map(Shaped, shapes))
)#";

  // Execute and get the fixed expectation value.
  auto locals = py::dict();
  py::exec(py_src, py::globals(), locals);
  auto eq = locals["eq"];
  auto arrays = locals["views"];
  auto opt_kwargs = pybind11::dict("max_repeats"_a = 16);
  auto optimizer = ctg.attr("HyperOptimizer")(opt_kwargs);
  // py::print(eq.attr("split")(",").attr("__len__")());
  // py::print(arrays.attr("__len__")());
  locals["optimizer"] = optimizer;
  locals["arrays"] = arrays;
  py::exec(
      R"#(contract_path = oe.contract_path(eq, *arrays, optimize=optimizer))#",
      py::globals(), locals);
  auto contract_path = locals["contract_path"];
  py::print(contract_path);
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
