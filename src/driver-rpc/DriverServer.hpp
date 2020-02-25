#ifndef EXATN_DRIVER_SERVER_HPP_
#define EXATN_DRIVER_SERVER_HPP_

#include "Identifiable.hpp"
#include "tensor_method.hpp"
#include "TAProLInterpreter.hpp"

#include <string>

namespace exatn {
namespace rpc {

class DriverServer : public Identifiable {

protected:

  std::shared_ptr<exatn::parser::TAProLInterpreter> parser;

public:

  DriverServer() = default;
  virtual void start() = 0;
  virtual void stop() = 0;
};
} // namespace rpc
} // namespace exatn
#endif
