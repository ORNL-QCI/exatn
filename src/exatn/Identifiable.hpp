#ifndef EXATN_IDENTIFIABLE_HPP_
#define EXATN_IDENTIFIABLE_HPP_

#include <string>

namespace exatn {

class Identifiable {
public:

  virtual const std::string name() const = 0;
  virtual const std::string description() const = 0;

  virtual ~Identifiable() {}
};

} // namespace exatn

#endif
