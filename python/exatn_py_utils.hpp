#ifndef EXATN_PY_UTILS_HPP_
#define EXATN_PY_UTILS_HPP_

#include "pybind11/detail/common.h"
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "pybind11/pybind11.h"
#include "talshxx.hpp"
#include "tensor_method.hpp"
#include "DriverClient.hpp"
#include "exatn.hpp"


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

using TensorFunctor = talsh::TensorFunctor<Identifiable>;

template <typename NumericType>
class NumpyTensorFunctorCppWrapper : public TensorFunctor {
protected:
  std::function<void(py::array_t<NumericType> &buffer)> _functor;
  py::array initialData;
  bool initialDataProvided = false;

public:
  NumpyTensorFunctorCppWrapper(
      std::function<void(py::array_t<NumericType> &buffer)> functor)
      : _functor(functor) {}
  NumpyTensorFunctorCppWrapper(py::array &buffer)
      : initialData(buffer), initialDataProvided(true) {}
  const std::string name() const override {
    return "numpy_tensor_functor_cpp_wrapper";
  }
  const std::string description() const override { return ""; }
  virtual void pack(BytePacket &packet) override {}
  virtual void unpack(BytePacket &packet) override {}

  int apply(talsh::Tensor &local_tensor) override {
    py::gil_scoped_release release;
    unsigned int nd = local_tensor.getRank();
    std::vector<std::size_t> dims_vec(nd);
    auto dims = local_tensor.getDimExtents(nd);
    for (int i = 0; i < nd; i++) {
      dims_vec[i] = dims[i];
    }

    NumericType *elements;
    auto worked =  local_tensor.getDataAccessHost(&elements);
    if (!worked) {
        std::stringstream ss;
        ss << "\nCould not get data of type " << typeid(NumericType).name() << "\n";
        std::cerr << ss.str() << "\n";
        assert(false);
    }

    if (initialDataProvided) {
      // If initial data is provided as a numpy array,
      // then I want to flatten it, and set it on the elements data
      std::vector<std::size_t> flattened{local_tensor.getVolume()};
      assert(local_tensor.getVolume() == initialData.size());
      initialData.resize(flattened);
      auto constelements =
          reinterpret_cast<const NumericType *>(initialData.data());
      for (int i = 0; i < local_tensor.getVolume(); i++) {
        elements[i] = constelements[i];
      }
    } else {
      auto cap = py::capsule(
          elements, [](void *v) { /* deleter, I do not own this... */ });
      py::gil_scoped_acquire acquire;
      auto arr = py::array_t<NumericType>(dims_vec, elements, cap);
      _functor(arr);
      py::gil_scoped_release r;
    }
    return 0;
  }
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

template <typename NumericType>
bool createTensorWithData(exatn::NumServer &n, const std::string name,
                          py::array_t<NumericType> &data) {
  auto shape = data.shape();
  std::vector<std::size_t> dims(data.ndim());
  for (int i = 0; i < data.ndim(); i++) {
    dims[i] = shape[i];
  }

  auto tensor_el_type = TypeToTensorElementType<NumericType>::type;
  auto created =
      n.createTensor(name, tensor_el_type, exatn::numerics::TensorShape(dims));
  assert(created);
  auto functor =
      std::make_shared<NumpyTensorFunctorCppWrapper<NumericType>>(data);
  return n.transformTensor(name, functor);
}

template <typename NumericType>
bool generalTransformWithData(
    exatn::NumServer &n, const std::string &name,
    std::function<void(py::array_t<NumericType> &buffer)> f) {
//   n.getTensorRef(name).
  auto functor = std::make_shared<NumpyTensorFunctorCppWrapper<NumericType>>(f);
  return n.transformTensor(name, functor);
}
}

#endif