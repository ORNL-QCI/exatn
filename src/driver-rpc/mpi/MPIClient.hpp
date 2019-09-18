#ifndef EXATN_MPICLIENT_HPP_
#define EXATN_MPICLIENT_HPP_

#include "DriverClient.hpp"
#include "DriverServer.hpp"
#include "mpi.h"
#include <algorithm>
#include <functional>
#include <map>
#include <iostream>


namespace exatn {
namespace rpc {
namespace mpi {

class MPIClient : public DriverClient {

private:
  std::function<char()> randChar = []() -> char {
    const char charset[] = "0123456789"
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                           "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[rand() % max_index];
  };

  const std::string generateRandomString(const int length = 10) {
    std::string str(length, 0);
    std::generate_n(str.begin(), length, randChar);
    return str;
  }

protected:

  MPI_Comm serverComm;
  std::map<std::string, MPI_Request> requests;
  std::map<std::string, int> jobId2NResults;
  std::vector<std::complex<double>> results;

  static int SYNC_TAG;
  static int SHUTDOWN_TAG;
  static int SENDTAPROL_TAG;
  static int REGISTER_TENSORMETHOD;

  bool connected = false;
  void connect();

public:

  MPIClient() = default;

  // Send TAProL code, get a jobId string, so this is a non-blocking asynchronous call.
  const std::string interpretTAProL(const std::string& taProlStr) override;

  // Retrieve results of a TAProL job with given jobId.
  // Currently retrieves saved complex<double> scalars.
  const std::vector<std::complex<double>> getResults(const std::string& jobId) override;

  // Register an external tensor method, a subclass of TensorFunctor class
  // which overrides the .apply(talsh::Tensor &) method. This allows
  // an application to initialize and transform tensors is a custom way
  // since the registered tensor methods will be accessible in TAProL text.
  void registerTensorMethod(const std::string& varName, talsh::TensorFunctor<Identifiable>& method) override;

  // Register external data under some symbolic name. This data will be accessible
  // in TAProL text. It can be used to define tensor dimensions dynamically, for example.
  void registerExternalData(const std::string& name, BytePacket& packet) override;

  // Shut down MPIClient.
  void shutdown() override;

  const std::string name() const override { return "mpi"; }
  const std::string description() const override {
    return "This DriverClient uses MPI to communicate with ExaTN Driver.";
  }
};

} // namespace mpi
} // namespace rpc
} // namespace exatn
#endif