#ifndef EXATN_PY_UTILS_HPP_
#define EXATN_PY_UTILS_HPP_

#include "num_server.hpp"
#include "pybind11/detail/common.h"
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "DriverClient.hpp"
#include "exatn.hpp"
#include "pybind11/pybind11.h"
#include "talshxx.hpp"
#include "tensor_basic.hpp"
#include "tensor_method.hpp"

#include <type_traits>

namespace py = pybind11;
using namespace exatn;
using namespace exatn::numerics;
using namespace pybind11::literals;

/**
  Trampoline class for abstract virtual functions in TensorOperation
*/

namespace exatn {
class PyTensorOperation : public exatn::numerics::TensorOperation {
public:
  /* Inherit the constructors */
  using TensorOperation::TensorOperation;
  void printIt() { PYBIND11_OVERLOAD(void, TensorOperation, printIt, ); }
  bool isSet() { PYBIND11_OVERLOAD_PURE(bool, TensorOperation, isSet, ); }
};

template <typename T> struct TypeToTensorElementType;
template <> struct TypeToTensorElementType<float> {
  static TensorElementType type;
  // = TensorElementType::COMPLEX32;
};
template <> struct TypeToTensorElementType<double> {
  static TensorElementType type;
};
template <> struct TypeToTensorElementType<std::complex<double>> {
  static TensorElementType type;
};
TensorElementType TypeToTensorElementType<float>::type =
    TensorElementType::REAL32;
TensorElementType TypeToTensorElementType<double>::type =
    TensorElementType::REAL64;
TensorElementType TypeToTensorElementType<std::complex<double>>::type =
    TensorElementType::COMPLEX64;

using TensorFunctor = talsh::TensorFunctor<Identifiable>;

class NumpyTensorFunctorCppWrapper : public TensorFunctor {
protected:
  std::function<void(py::array& buffer)> _functor;
  py::array initialData;
  bool initialDataProvided = false;
  TensorElementType tensorType;

  template <typename NumericType>
  void setTensorData(NumericType *elements,
                     const std::vector<std::size_t> dims_vec,
                     const std::size_t volume) {

    if (initialDataProvided) {
      // If initial data is provided as a numpy array,
      // then I want to flatten it, and set it on the elements data
      std::vector<std::size_t> flattened{volume};
      assert(volume == initialData.size());
      initialData.resize(flattened);
      auto constelements =
          reinterpret_cast<const NumericType *>(initialData.data());
      for (int i = 0; i < volume; i++) {
        elements[i] = constelements[i];
      }
    } else {
      auto cap = py::capsule(
          elements, [](void *v) { /* deleter, I do not own this... */ });
      auto arr = py::array(dims_vec, elements, cap);
      _functor(arr);
    }
  }

public:
  NumpyTensorFunctorCppWrapper(std::function<void(py::array& buffer)> &functor,
                               TensorElementType type)
      : _functor(functor), tensorType(type) {}
  NumpyTensorFunctorCppWrapper(py::array buffer, TensorElementType type)
      : initialData(buffer), initialDataProvided(true), tensorType(type) {}
  const std::string name() const override {
    return "numpy_tensor_functor_cpp_wrapper";
  }
  const std::string description() const override { return ""; }
  virtual void pack(BytePacket &packet) override {}
  virtual void unpack(BytePacket &packet) override {}

  int apply(talsh::Tensor &local_tensor) override {
    auto volume = local_tensor.getVolume();
    unsigned int nd = local_tensor.getRank();
    std::vector<std::size_t> dims_vec(nd);
    auto dims = local_tensor.getDimExtents(nd);
    for (int i = 0; i < nd; i++) {
      dims_vec[i] = dims[i];
    }

    if (tensorType == TensorElementType::REAL64) {
      double *elements;
      auto worked = local_tensor.getDataAccessHost(&elements);
      setTensorData<double>(elements, dims_vec, volume);
    } else if (tensorType == TensorElementType::COMPLEX64) {
      std::complex<double> *elements;
      auto worked = local_tensor.getDataAccessHost(&elements);
      setTensorData<std::complex<double>>(elements, dims_vec, volume);
    } else if (tensorType == TensorElementType::COMPLEX32) {
      std::complex<float> *elements;
      auto worked = local_tensor.getDataAccessHost(&elements);
      setTensorData<std::complex<float>>(elements, dims_vec, volume);
    } else if (tensorType == TensorElementType::REAL32) {
      float *elements;
      auto worked = local_tensor.getDataAccessHost(&elements);
      setTensorData<float>(elements, dims_vec, volume);
    } else {
      assert(false && "Invalid TensorElementType");
    }

    return 0;
  }
};


bool createTensorWithDataNoNumServer(const std::string name,
                          py::array &data) {
  auto n = exatn::numericalServer;
  auto shape = data.shape();
  std::vector<std::size_t> dims(data.ndim());
  for (int i = 0; i < data.ndim(); i++) {
    dims[i] = shape[i];
  }

  // Learn underlying data type of py::array in order
  // to set TensorElementType
  TensorElementType type;
  if (py::isinstance<py::array_t<double>>(data)) {
    type = TensorElementType::REAL64;
  } else if (py::isinstance<py::array_t<std::complex<double>>>(data)) {
    type = TensorElementType::COMPLEX64;
  } else if (py::isinstance<py::array_t<float>>(data)) {
    type = TensorElementType::REAL32;
  } else if (py::isinstance<py::array_t<std::complex<float>>>(data)) {
    type = TensorElementType::COMPLEX32;
  }

  auto created = n->createTensor(name, type, exatn::numerics::TensorShape(dims));
  assert(created);
  auto functor = std::make_shared<NumpyTensorFunctorCppWrapper>(data, type);
  return n->transformTensorSync(name, functor);
}

bool createTensorWithData(exatn::NumServer &n, const std::string name,
                          py::array &data) {
  auto shape = data.shape();
  std::vector<std::size_t> dims(data.ndim());
  for (int i = 0; i < data.ndim(); i++) {
    dims[i] = shape[i];
  }

  // Learn underlying data type of py::array in order
  // to set TensorElementType
  TensorElementType type;
  if (py::isinstance<py::array_t<double>>(data)) {
    type = TensorElementType::REAL64;
  } else if (py::isinstance<py::array_t<std::complex<double>>>(data)) {
    type = TensorElementType::COMPLEX64;
  } else if (py::isinstance<py::array_t<float>>(data)) {
    type = TensorElementType::REAL32;
  } else if (py::isinstance<py::array_t<std::complex<float>>>(data)) {
    type = TensorElementType::COMPLEX32;
  }

  auto created = n.createTensor(name, type, exatn::numerics::TensorShape(dims));
  assert(created);
  auto functor = std::make_shared<NumpyTensorFunctorCppWrapper>(data, type);
  return n.transformTensorSync(name, functor);
}

bool generalTransformWithData(exatn::NumServer &n, const std::string &name,
                              std::function<void(py::array& buffer)> &f) {
  auto type = n.getTensorElementType(name);
  std::function<void(py::array&)> wrapper = [f] (py::array& buffer) {
      py::gil_scoped_release r;
      f(buffer);
      return;
  };
  auto functor = std::make_shared<NumpyTensorFunctorCppWrapper>(
      wrapper,
      type);
  auto worked = n.transformTensorSync(name, functor);
  return worked;
}


bool generalTransformWithDataNoNumServer(const std::string &name,
                              std::function<void(py::array& buffer)> &f) {
auto n = exatn::numericalServer;
  auto type = n->getTensorElementType(name);
  std::function<void(py::array&)> wrapper = [f] (py::array& buffer) {
      py::gil_scoped_release r;
      f(buffer);
      return;
  };
  auto functor = std::make_shared<NumpyTensorFunctorCppWrapper>(
      wrapper,
      type);
  auto worked = n->transformTensorSync(name, functor);
  return worked;
}

void printTensorData(exatn::NumServer &n, const std::string &name) {
  auto type = n.getTensorElementType(name);
  std::function<void(py::array&)> f = [](py::array& data) {
     py::print("Shape: ", data.shape());
     py::print(data);

  };
  auto functor = std::make_shared<NumpyTensorFunctorCppWrapper>(
      f,
      type);
  auto worked = n.transformTensorSync(name, functor);
  return;
}


void printTensorDataNoNumServer(const std::string &name) {
    auto n = exatn::numericalServer;
  auto type = n->getTensorElementType(name);
  std::function<void(py::array&)> f = [](py::array& data) {
     py::print("Shape: ", data.shape());
     py::print(data);

  };
  auto functor = std::make_shared<NumpyTensorFunctorCppWrapper>(
      f,
      type);
  auto worked = n->transformTensorSync(name, functor);
  return;
}
} // namespace exatn

#endif