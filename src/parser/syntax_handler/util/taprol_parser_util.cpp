#include "taprol_parser_util.hpp"
#include "TAProLInterpreter.hpp"

namespace exatn {
namespace container {
template <typename ContainerType, typename ElementType>
bool contains(const ContainerType &container, const ElementType &item) {
  return std::find(container.begin(), container.end(), item) != container.end();
}
} // namespace container
std::string run_token_collector(clang::Preprocessor &PP,
                                clang::CachedTokens &Toks,
                                std::map<std::string, std::string> &args) {

  // Merge Toks into
  std::stringstream ss;
  for (int i = 0; i < Toks.size(); i++) {
    ss << PP.getSpelling(Toks[i]);
    if (container::contains(std::vector<std::string>{"scope", "main", "end", "save", "destroy"},
                            PP.getSpelling(Toks[i]))) {
      ss << " ";
    }
  }

  std::stringstream new_code_ss;
  parser::TAProLInterpreter interpreter;
  interpreter.interpret(ss.str(), new_code_ss, args);
  std::cout << new_code_ss.str() << "\n";

  return new_code_ss.str();
}

} // namespace exatn
