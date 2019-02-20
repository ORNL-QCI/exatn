
#include "exatn.hpp"
#include "DriverClient.hpp"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>

namespace py = pybind11;
using namespace exatn;

PYBIND11_MODULE(_pyexatn, m) {
  m.doc() =
      "Python bindings for ExaTN.";

  py::class_<BytePacket>(
      m, "BytePacket", "")
      .def_readwrite("base_addr", &BytePacket::base_addr,"")
      .def_readwrite("size_bytes", &BytePacket::size_bytes, "");

  py::class_<TensorDenseBlock>(
      m, "TensorDenseBlock", "")
      .def_readwrite("num_dims", &TensorDenseBlock::num_dims,"")
      .def_readwrite("data_kind", &TensorDenseBlock::data_kind, "")
      .def_readwrite("body_ptr", &TensorDenseBlock::body_ptr, "")
      .def_readwrite("bases", &TensorDenseBlock::bases, "")
      .def_readwrite("dims", &TensorDenseBlock::dims, "");

  py::class_<TensorMethod<Identifiable>, std::shared_ptr<TensorMethod<Identifiable>>>(
      m, "TensorMethod", "")
      .def("pack", &TensorMethod<Identifiable>::pack, "")
      .def("unpack", &TensorMethod<Identifiable>::unpack, "")
      .def("apply", &TensorMethod<Identifiable>::apply, "");

  py::class_<exatn::rpc::DriverClient, std::shared_ptr<exatn::rpc::DriverClient>>(
      m, "DriverClient","")
      .def("interpretTAProL", &exatn::rpc::DriverClient::interpretTAProL, "")
      .def("registerTensorMethod", &exatn::rpc::DriverClient::registerTensorMethod, "")
      .def("getResults", &exatn::rpc::DriverClient::getResults, "")
      .def("shutdown", &exatn::rpc::DriverClient::shutdown, "");

  m.def("Initialize", (void (*)()) & exatn::Initialize,
        "Initialize the exatn framework.");
  m.def("getDriverClient",[](const std::string name) -> std::shared_ptr<exatn::rpc::DriverClient> {
      return exatn::getService<exatn::rpc::DriverClient>(name);
      }, "");
  m.def("Finalize", &exatn::Finalize, "Finalize the framework");

}
