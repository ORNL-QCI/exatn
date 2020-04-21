#include "exatn_py_utils.hpp"
#include "pybind11/pybind11.h"
#include "talshxx.hpp"
#include "tensor_basic.hpp"

#include "exatn_numerics.hpp"
#include "tensor_expansion.hpp"
#include "tensor_network.hpp"
#include "tensor_operation.hpp"
#include "tensor_operator.hpp"

namespace py = pybind11;
using namespace exatn;
using namespace exatn::numerics;
using namespace pybind11::literals;

namespace exatn {

void create_exatn_py_module(py::module &m) {
  m.doc() = "Python bindings for ExaTN.";
  py::class_<exatn::numerics::Tensor, std::shared_ptr<exatn::numerics::Tensor>>(
      m, "Tensor", "")
      .def("printIt", &exatn::numerics::Tensor::printIt, "with_hash"_a = false, "")
      .def("getName", &exatn::numerics::Tensor::getName, "")
      .def("getRank", &exatn::numerics::Tensor::getRank, "");

  py::class_<TensorExpansion::ExpansionComponent, std::shared_ptr<TensorExpansion::ExpansionComponent>>(m, "ExpansionComponent", "")
     .def_readonly("network", &TensorExpansion::ExpansionComponent::network_,"");

  py::class_<exatn::numerics::TensorOperator,
             std::shared_ptr<exatn::numerics::TensorOperator>>(
      m, "TensorOperator", "")
      .def(py::init<std::string>())
      .def(
          "appendComponent",
          [](TensorOperator &_operator, std::shared_ptr<Tensor> tensor,
             const std::vector<std::pair<unsigned int, unsigned int>>
                 &ket_pairing,
             const std::vector<std::pair<unsigned int, unsigned int>>
                 &bra_pairing,
             const std::complex<double> coefficient) {
            return _operator.appendComponent(tensor, ket_pairing, bra_pairing,
                                             coefficient);
          },
          "");
py::class_<exatn::numerics::TensorExpansion,
             std::shared_ptr<exatn::numerics::TensorExpansion>>(m, "TensorExpansion", "")
    .def(py::init<>())
    .def(py::init<const TensorExpansion&>())
    .def(py::init<const TensorExpansion&, const TensorOperator&>())
    .def(py::init<const TensorExpansion&, const TensorExpansion&>())
    .def(py::init<const TensorExpansion&, const TensorExpansion&, const TensorOperator&>())
    .def(py::init<const TensorExpansion&, const std::string&, bool>())
    .def("appendComponent", &TensorExpansion::appendComponent, "")
    .def("conjugate", &TensorExpansion::conjugate, "")
    .def("printIt", &TensorExpansion::printIt, "")
    .def("rename", &TensorExpansion::rename, "")
    .def("__iter__", [](const TensorExpansion &s) { return py::make_iterator(s.cbegin(), s.cend()); },
                         py::keep_alive<0, 1>());

  py::class_<exatn::numerics::TensorNetwork,
             std::shared_ptr<exatn::numerics::TensorNetwork>>(
      m, "TensorNetwork", "")
      .def(py::init<>())
      .def(py::init<const std::string>())
      .def(py::init<const TensorNetwork &>())
      .def(py::init<const std::string &, const std::string &,
                    const std::map<std::string, std::shared_ptr<Tensor>>>())
      .def("printIt", &exatn::numerics::TensorNetwork::printIt, "with_tensor_hash"_a = false, "")
      .def("rename", &exatn::numerics::TensorNetwork::rename, "")
      .def("getName", &exatn::numerics::TensorNetwork::getName, "")
      .def("conjugate", &TensorNetwork::conjugate, "")
      .def("getRank", &TensorNetwork::getRank, "")
      .def("isEmpty", &exatn::numerics::TensorNetwork::isEmpty, "")
      .def("isExplicit", &exatn::numerics::TensorNetwork::isExplicit, "")
      .def("finalize", &exatn::numerics::TensorNetwork::finalize,
           "check_validity"_a = false, "")
      .def("isFinalized", &exatn::numerics::TensorNetwork::isFinalized, "")
      .def("getNumTensors", &exatn::numerics::TensorNetwork::getNumTensors, "")
      .def(
          "getTensor",
          [](TensorNetwork &network, const unsigned int id) {
            return network.getTensor(id);
          },
          "")
      .def("collapseIsometries", &TensorNetwork::collapseIsometries, "")
      .def("getTensor", &TensorNetwork::getTensor, "")
      .def("placeTensor",
           (bool (exatn::numerics::TensorNetwork::*)(
               unsigned int, std::shared_ptr<exatn::numerics::Tensor>,
               const std::vector<exatn::numerics::TensorLeg> &, bool conjugated,
               bool leg_matching_check)) &
               exatn::numerics::TensorNetwork::placeTensor,
           "")
      .def(
          "appendTensor",
          [](TensorNetwork &network, unsigned int tensor_id,
             const std::string &name) {
            if (!network.appendTensor(tensor_id, exatn::getTensor(name), {})) {
              std::cerr << "Could not append " << tensor_id << ", " << name
                        << "\n";
              exit(1);
            }
          },
          "")
      .def(
          "appendTensor",
          [](TensorNetwork &network, unsigned int tensor_id,
             const std::string &name,
             const std::vector<std::pair<unsigned int, unsigned int>>
                 &pairing) {
            if (!network.appendTensor(tensor_id, exatn::getTensor(name),
                                      pairing)) {
              std::cerr << "Could not append " << tensor_id << ", " << name
                        << "\n";
              exit(1);
            }
          },
          "")
      .def(
          "appendTensorGate",
          [](TensorNetwork &network, unsigned int tensor_id,
             const std::string &name,
             const std::vector<unsigned int> leg_pairing) {
            if (!network.appendTensorGate(tensor_id, exatn::getTensor(name),
                                          leg_pairing)) {
              std::cerr << "Could not append gate " << tensor_id << ", " << name
                        << "\n";
              exit(1);
            }
          },
          "")
      .def(
          "appendTensorGate",
          [](TensorNetwork &network, unsigned int tensor_id,
             const std::string &name,
             const std::vector<unsigned int> leg_pairing, bool conjugate) {
            if (!network.appendTensorGate(tensor_id, exatn::getTensor(name),
                                          leg_pairing, conjugate)) {
              std::cerr << "Could not append gate " << tensor_id << ", " << name
                        << "\n";
              exit(1);
            }
          },
          "")
      .def(
          "appendTensorNetwork",
          [](TensorNetwork &network, TensorNetwork &otherNetwork,
             const std::vector<std::pair<unsigned int, unsigned int>>
                 &pairing) {
            return network.appendTensorNetwork(std::move(otherNetwork),
                                               pairing);
          },
          "")
      .def("reorderOutputModes",
           &exatn::numerics::TensorNetwork::reorderOutputModes, "")
      .def("deleteTensor", &exatn::numerics::TensorNetwork::deleteTensor, "")
      .def("mergeTensors", &exatn::numerics::TensorNetwork::mergeTensors, "");

  py::enum_<exatn::TensorElementType>(m, "DataType", py::arithmetic(), "")
      .value("float32", exatn::TensorElementType::REAL32, "")
      .value("float64", exatn::TensorElementType::REAL64, "")
      .value("complex32", exatn::TensorElementType::COMPLEX32, "")
      .value("complex64", exatn::TensorElementType::COMPLEX64, "")
      .value("complex", exatn::TensorElementType::COMPLEX64, "")
      .value("float", exatn::TensorElementType::REAL64, "");

  /**
   ExaTN module definitions
  */
  m.def(
      "Initialize", []() { return exatn::initialize(); },
      "Initialize the exatn framework.");
#ifdef MPI_ENABLED
  m.def(
      "Initialize",
      [](const exatn::MPICommProxy &communicator) {
        return exatn::initialize(communicator);
      },
      "Initialize the exatn framework.");
#endif

  m.def("Finalize", &exatn::finalize, "Finalize the framework.");

  // TensorNetwork *network is the network to append to, and TensorNetwork
  // append_network is the network that will be appended to *network PyBind
  // cannot bind this function simply within the TensorNetwork class due to the
  // && argument
  m.def(
      "appendTensorNetwork",
      [](TensorNetwork *network, TensorNetwork append_network,
         const std::vector<std::pair<unsigned int, unsigned int>> &pairing) {
        return network->appendTensorNetwork(std::move(append_network), pairing);
      },
      "");

  m.def("createTensor", [](const std::string &name, double &value) {
    auto success = exatn::createTensor(name, exatn::TensorElementType::REAL64);
    if (success) {
      return exatn::initTensorSync(name, value);
    }
    return success;
  });
  m.def("createTensor",
        [](const std::string &name, std::complex<double> &value) {
          auto success =
              exatn::createTensor(name, exatn::TensorElementType::COMPLEX64);
          if (success) {
            return exatn::initTensorSync(name, value);
          }
          return success;
        });

  m.def(
      "createTensor",
      [](const std::string &name, TensorElementType type) {
        return exatn::createTensor(name, type);
      },
      "");
  m.def(
      "createTensor",
      [](const std::string &name, std::vector<std::size_t> dims,
         TensorElementType type) {
        return exatn::createTensor(name, type,
                                   exatn::numerics::TensorShape(dims));
      },
      "");
  m.def(
      "createTensor",
      [](const std::string &name, std::vector<std::size_t> dims,
         double &init_value) {
        auto success =
            exatn::createTensor(name, exatn::TensorElementType::REAL64,
                                exatn::numerics::TensorShape(dims));
        if (success) {
          return exatn::initTensorSync(name, init_value);
        }
        return success;
      },
      "");
 m.def(
      "createTensor",
      [](const std::string &name, std::vector<std::size_t> dims,
         std::complex<double> &init_value) {
        auto success =
            exatn::createTensor(name, exatn::TensorElementType::COMPLEX64,
                                exatn::numerics::TensorShape(dims));
        if (success) {
          return exatn::initTensorSync(name, init_value);
        }
        return success;
      },
      "");
  m.def(
      "createTensor",
      [](const std::string &name) {
        auto success =
            exatn::createTensor(name, exatn::TensorElementType::REAL64);
        if (success) {
          return exatn::initTensorSync(name, 0.0);
        }
        return success;
      },
      "");
  m.def("createTensor", &createTensorWithDataNoNumServer, "");
  m.def(
      "registerTensorIsometry",
      [](const std::string &name, const std::vector<unsigned int> &iso_dims) {
        return exatn::registerTensorIsometry(name, iso_dims);
      },
      "");
  m.def(
      "registerTensorIsometry",
      [](const std::string &name, const std::vector<unsigned int> &iso_dims0,
         const std::vector<unsigned int> &iso_dims1) {
        return exatn::registerTensorIsometry(name, iso_dims0, iso_dims1);
      },
      "");
  m.def(
      "evaluate",
      [](TensorNetwork &network) { return evaluateSync(network); },
      "");
  m.def(
      "evaluate",
      [](TensorExpansion& exp, std::shared_ptr<Tensor> accum){return exatn::evaluateSync(exp,accum);});
  m.def("getTensor", &exatn::getTensor, "");
  m.def("print", &printTensorDataNoNumServer, "");
  m.def("transformTensor", &generalTransformWithDataNoNumServer, "");
  m.def(
      "evaluateTensorNetwork",
      [](const std::string& name, const std::string& network){
         return exatn::evaluateTensorNetworkSync(name,network);},
      "");
  m.def(
      "evaluateTensorNetwork",
      [](const std::string& name, const std::string& network, const ProcessGroup & process_group){
         return exatn::evaluateTensorNetworkSync(name,network,process_group);},
      "");
  m.def("getTensorData", &getTensorData, "");
  m.def("getLocalTensor", [](const std::string &name) {
    auto local_tensor = exatn::getLocalTensor(name);
    unsigned int nd = local_tensor->getRank();

    std::vector<std::size_t> dims_vec(nd);
    auto dims = local_tensor->getDimExtents(nd);
    for (int i = 0; i < nd; i++) {
      dims_vec[i] = dims[i];
    }

    auto tensorType = local_tensor->getElementType();

    if (tensorType == talsh::REAL64) {
      double *elements;
      auto worked = local_tensor->getDataAccessHost(&elements);
      auto cap = py::capsule(
          elements, [](void *v) { /* deleter, I do not own this... */ });
      auto arr = py::array(dims_vec, elements, cap);
      return arr;
    } else if (tensorType == talsh::COMPLEX64) {
      std::complex<double> *elements;
      auto worked = local_tensor->getDataAccessHost(&elements);
      auto cap = py::capsule(
          elements, [](void *v) { /* deleter, I do not own this... */ });
      auto arr = py::array(dims_vec, elements, cap);
      return arr;

    } else if (tensorType == talsh::COMPLEX32) {
      std::complex<float> *elements;
      auto worked = local_tensor->getDataAccessHost(&elements);
      auto cap = py::capsule(
          elements, [](void *v) { /* deleter, I do not own this... */ });
      auto arr = py::array(dims_vec, elements, cap);
      return arr;

    } else if (tensorType == talsh::REAL32) {
      float *elements;
      auto worked = local_tensor->getDataAccessHost(&elements);
      auto cap = py::capsule(
          elements, [](void *v) { /* deleter, I do not own this... */ });
      auto arr = py::array(dims_vec, elements, cap);
      return arr;

    } else {
      assert(false && "Invalid TensorElementType");
    }
    return py::array();
  });
  m.def("destroyTensor", &destroyTensor, "");
}
} // namespace exatn

PYBIND11_MODULE(_pyexatn, m) { exatn::create_exatn_py_module(m); }
