#ifndef EXATN_DRIVER_CLIENT_HPP_
#define EXATN_DRIVER_CLIENT_HPP_

#include <string>
#include "Identifiable.hpp"

namespace exatn {

class DriverClient : public Identifiable {

public:

   // Send TaProl string, get a jobId string,
   // so this is an asynchronous call
   virtual const std::string sendTaProl(const std::string taProlStr) = 0;

   // Retrieve result of job with given jobId.
   // Returns a scalar type double?
   virtual const double retrieveResult(const std::string jobId) = 0;

};

}
#endif