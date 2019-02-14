
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

  py::class_<exatn::rpc::DriverClient, std::shared_ptr<exatn::rpc::DriverClient>>(
      m, "DriverClient","")
      .def("sendTAProL", &exatn::rpc::DriverClient::sendTAProL, "")
      .def("retrieveResults", &exatn::rpc::DriverClient::retrieveResult, "")
      .def("shutdown", &exatn::rpc::DriverClient::shutdown, "");


  m.def("Initialize", (void (*)()) & exatn::Initialize,
        "Initialize the exatn framework.");
  m.def("getDriverClient",[](const std::string name) -> std::shared_ptr<exatn::rpc::DriverClient> {
      return exatn::getService<exatn::rpc::DriverClient>(name);
      }, "");
  m.def("Finalize", &exatn::Finalize, "Finalize the framework");

}
