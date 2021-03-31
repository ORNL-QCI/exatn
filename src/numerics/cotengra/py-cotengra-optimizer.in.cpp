#include "contraction_seq_optimizer.hpp"
#include "contraction_seq_optimizer_cotengra.hpp"
#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "tensor_network.hpp"
#include <dlfcn.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
using namespace cppmicroservices;
using namespace pybind11::literals;
namespace py = pybind11;

namespace exatn {
namespace numerics {
double ContractionSeqOptimizerCotengra::determineContractionSequence(
    const TensorNetwork &network, std::list<ContrTriple> &contr_seq,
    std::function<unsigned int()> intermediate_num_generator,
    uint64_t target_slice_size, std::list<SliceIndex> &slice_inds) {
  static bool initialized = false;
  static std::shared_ptr<pybind11::scoped_interpreter> guard;
  static void *libpython_handle = nullptr;

  if (!initialized) {
    guard = std::make_shared<py::scoped_interpreter>();
    libpython_handle = dlopen("@PYTHON_LIB_NAME@", RTLD_LAZY | RTLD_GLOBAL);
    initialized = true;
  }

  const auto time_beg = std::chrono::high_resolution_clock::now();
  // network.printIt();
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
    tensor_rank_map[py::int_(tensorId)] = tensor_conn.getDimExtents();
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
  globals["ctg"] = ctg;
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
  shape_map[tensor_id] = tuple(rank_data[tensor_id])

from opt_einsum.contract import Shaped
inputs = []
output = {}
node_list = []
for nd in graph.nodes:
  if nd == 0:
    output = {edge2ind[tuple(sorted(e))] for e in graph.edges(nd)}
  else:
    inputs.append({edge2ind[tuple(sorted(e))] for e in graph.edges(nd)})
    node_list.append(int(nd))

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
  locals["target_size"] = target_slice_size;
  // Cotengra number of iterations
  const int max_repeats = 1000;
  locals["max_repeats"] = max_repeats;

  // Create Cotengra HyperOptimizer:
  // Note: this config is for Sycamore circuits.
  py::exec(R"#(
opt = ctg.HyperOptimizer(slicing_reconf_opts={'target_size': locals()['target_size']},  max_repeats=locals()['max_repeats'], parallel='auto', progbar=True))#",
           py::globals(), locals);

  // auto optimizer = ctg.attr("HyperOptimizer")();
  auto optimizer = locals["opt"];
  locals["optimizer"] = optimizer;
  locals["arrays"] = arrays;
  py::exec(
      R"#(path_list, path_info = oe.contract_path(eq, *arrays, optimize=optimizer))#",
      py::globals(), locals);
  auto contract_path = locals["path_list"];
  // py::print(contract_path);
  auto path_vec = contract_path.cast<std::vector<std::pair<int, int>>>();
  auto node_list = locals["node_list"].cast<std::vector<int>>();

  const auto remove_index = [](std::vector<int> &vector,
                               const std::vector<int> &to_remove) {
    auto vector_base = vector.begin();
    std::vector<int>::size_type down_by = 0;

    for (auto iter = to_remove.cbegin(); iter < to_remove.cend();
         iter++, down_by++) {
      std::vector<int>::size_type next =
          (iter + 1 == to_remove.cend() ? vector.size() : *(iter + 1));

      std::move(vector_base + *iter + 1, vector_base + next,
                vector_base + *iter - down_by);
    }
    vector.resize(vector.size() - to_remove.size());
  };

  contr_seq.clear();
  for (const auto &path_pair : path_vec) {
    const auto lhs_node = node_list[path_pair.first];
    const auto rhs_node = node_list[path_pair.second];
    // Remove these 2 nodes:
    remove_index(node_list, {path_pair.first, path_pair.second});
    const auto intermediate_tensor_id = intermediate_num_generator();
    // std::cout << "Contract: " << lhs_node << " and " << rhs_node << " --> "
    //           << intermediate_tensor_id << "\n";
    node_list.emplace_back(intermediate_tensor_id);
    ContrTriple contrTriple;
    contrTriple.result_id = intermediate_tensor_id;
    contrTriple.left_id = lhs_node;
    contrTriple.right_id = rhs_node;
    contr_seq.emplace_back(contrTriple);
  }

  auto &lastSeq = contr_seq.back();
  lastSeq.result_id = 0;

  auto tree = optimizer.attr("best")["tree"];
  // py::print(tree);
  const double flops = tree.attr("contraction_cost")().cast<double>();
  // std::cout << "Contraction cost: " << flops << "\n";
  const auto time_end = std::chrono::high_resolution_clock::now();
  const auto time_total =
      std::chrono::duration_cast<std::chrono::duration<double>>(time_end -
                                                                time_beg);
  std::cout << "#DEBUG(ContractionSeqOptimizerCotengra): Done ("
            << time_total.count() << " sec): " << flops << " flops.\n";

  auto slice_ids = tree.attr("sliced_inds");
  py::print(slice_ids);
  auto iter = py::iter(slice_ids);
  std::vector<std::pair<int, int>> slice_edges;
  while (iter != py::iterator::sentinel()) {
    locals["sliceIdx"] = *iter;
    py::exec(
        R"#(slice_edge = list(edge2ind.keys())[list(edge2ind.values()).index(locals()['sliceIdx'])])#",
        py::globals(), locals);
    const auto slice_edge = locals["slice_edge"].cast<std::pair<int, int>>();
    std::cout << slice_edge.first << ":" << slice_edge.second << "\n";
    slice_edges.emplace_back(slice_edge);
    ++iter;
  }

  for (const auto &slice_edge : slice_edges) {
    const int lhs_tensor_id = slice_edge.first;
    const int rhs_tensor_id = slice_edge.second;
    const auto lhsTensor = network.getTensor(lhs_tensor_id);
    const auto rhsTensor = network.getTensor(rhs_tensor_id);
    // lhsTensor->printIt();
    // std::cout << "\n";
    // rhsTensor->printIt();

    const auto lhsTensorLegs = *(network.getTensorConnections(lhs_tensor_id));
    const auto rhsTensorLegs = *(network.getTensorConnections(rhs_tensor_id));
    bool foundLeg = false;
    for (const auto &lhsTensorLeg : lhsTensorLegs) {
      if (lhsTensorLeg.getTensorId() == rhs_tensor_id) {
        foundLeg = true;
        SliceIndex slice_index;
        slice_index.tensor_id = lhsTensorLeg.getTensorId();
        slice_index.leg_id = lhsTensorLeg.getDimensionId();
        slice_inds.emplace_back(slice_index);
        // std::cout << "Tensor ID: " << slice_index.tensor_id
        //           << "; Leg: " << slice_index.leg_id << "\n";
        break;
      }
    }
    assert(foundLeg);
    foundLeg = false;
    for (const auto &rhsTensorLeg : rhsTensorLegs) {
      if (rhsTensorLeg.getTensorId() == lhs_tensor_id) {
        foundLeg = true;
        SliceIndex slice_index;
        slice_index.tensor_id = rhsTensorLeg.getTensorId();
        slice_index.leg_id = rhsTensorLeg.getDimensionId();
        slice_inds.emplace_back(slice_index);
        // std::cout << "Tensor ID: " << slice_index.tensor_id
        //           << "; Leg: " << slice_index.leg_id << "\n";
        break;
      }
    }
    assert(foundLeg);
  }

  return flops;
}

std::unique_ptr<ContractionSeqOptimizer>
ContractionSeqOptimizerCotengra::createNew() {
  return std::make_unique<ContractionSeqOptimizerCotengra>();
}
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
