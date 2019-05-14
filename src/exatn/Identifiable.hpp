#ifndef EXATN_IDENTIFIABLE_HPP_
#define EXATN_IDENTIFIABLE_HPP_

#include <string>
#include <memory>

namespace exatn {

class Identifiable {
public:

  virtual const std::string name() const = 0;
  virtual const std::string description() const = 0;

  virtual ~Identifiable() {}
};

template <typename T> class Cloneable {
public:
  virtual std::shared_ptr<T> clone() = 0;

  /**
   * The destructor
   */
  virtual ~Cloneable() {}
};

} // namespace exatn

#endif
