#ifndef EXATN_DRIVER_SERVER_HPP_
#define EXATN_DRIVER_SERVER_HPP_

#include "Identifiable.hpp"
#include <string>

namespace exatn {
namespace rpc {
class DriverServer : public Identifiable {

public:
  virtual void start() = 0;

  virtual void stop() = 0;
};
} // namespace rpc
} // namespace exatn
#endif