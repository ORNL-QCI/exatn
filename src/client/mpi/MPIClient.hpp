#ifndef EXATN_MPIDRIVER_CLIENT_HPP_
#define EXATN_MPIDRIVER_CLIENT_HPP_

#include <string>
#include "DriverClient.hpp"

namespace exatn {


class MPIDriverClient : public DriverClient {

public:

   // Send TaProl string, get a jobId string,
   // so this is an asynchronous call
   const std::string sendTaProl(const std::string taProlStr) override {

   }

   // Retrieve result of job with given jobId.
   // Returns a scalar type double?
   const double retrieveResult(const std::string jobId) override {

   }

  const std::string name() const override {
      return "mpi";
  }
  
  const std::string description() const override {
      return "This DriverClient uses MPI to communicate with ExaTN Driver.";
  }

};

}
#endif