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

  // Send TAProL code, get a jobId string, so this is a non-blocking asynchronous call.
  virtual const std::string interpretTAProL(const std::string& taProlStr) = 0;

  // Retrieve results of a TAProL job with given jobId.
  // Currently retrieves saved complex<double> scalars.
  virtual const std::vector<std::complex<double>> getResults(const std::string& jobId) = 0;

  // Register an external tensor method, a subclass of TensorMethod class
  // which overrides the .apply(const TensorDenseBlock &) method. This
  // allows an application to initialize and transform tensors is a custom way
  // since the registered tensor methods will be accessible in TAProL text.
  virtual void registerTensorMethod(TensorMethod<Identifiable>& method) = 0;

  // Register external data under some symbolic name. This data will be accessible
  // in TAProL text. It can be used to define tensor dimensions dynamically, for example.
  virtual void registerExternalData(const std::string& name, BytePacket& packet) = 0;

  // Shut down DriverClient.
  virtual void shutdown() = 0;

};
} // namespace rpc
} // namespace exatn
#endif
