#ifndef EXATN_DRIVER_CLIENT_HPP_
#define EXATN_DRIVER_CLIENT_HPP_

#include "Identifiable.hpp"
#include <string>
#include <vector>
#include <complex>
#include "tensor_method.hpp"

namespace exatn {
namespace rpc {

class DriverClient : public Identifiable {

public:

  // Send TaProl string, get a jobId string,
  // so this is an asynchronous call
  virtual const std::string interpretTAProL(const std::string& taProlStr) = 0;

  // Retrieve result of job with given jobId.
  // Returns a scalar type double?
  virtual const std::vector<std::complex<double>> getResults(const std::string& jobId) = 0;

  virtual void registerTensorMethod(TensorMethod<Identifiable>& method) = 0;

  virtual void shutdown() = 0;

};
} // namespace rpc
} // namespace exatn
#endif