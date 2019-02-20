#ifndef EXATN_NUMERICS_TALSH_BACKEND_HPP_
#define EXATN_NUMERICS_TALSH_BACKEND_HPP_

#include "backend.hpp"
#include <string>
#include <vector>
#include <functional>

namespace exatn {
namespace numerics {
namespace talsh {

class TalshBackend : public Backend {

protected:
  std::function<std::vector<std::string>(const std::string &,
                                         const std::string &)>
      split =
          [](const std::string &str,
             const std::string &delimiter) -> std::vector<std::string> {
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos) {
      strings.push_back(str.substr(prev, pos - prev));
      prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
  };

public:

  std::vector<std::string> translate(const std::string taProl) override;
  void execute(std::vector<std::string> &simpleTaPrlList) override;

  const std::string name() const override { return "talsh"; }
  const std::string description() const override { return ""; }
};

} // namespace talsh
} // namespace numerics
} // namespace exatn

#endif