#ifndef EXATN_NUMERICS_BACKEND_HPP_
#define EXATN_NUMERICS_BACKEND_HPP_

#include "Identifiable.hpp"
#include <string>
#include <vector>

namespace exatn {
namespace numerics {

class Backend : public Identifiable {

public:

  virtual std::vector<std::string> translate(const std::string taProl) = 0;
  virtual void execute(std::vector<std::string>& simpleTaPrlList) = 0;

};

}
}

#endif