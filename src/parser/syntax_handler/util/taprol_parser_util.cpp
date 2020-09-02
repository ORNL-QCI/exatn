#include "taprol_parser_util.hpp"
#include "TAProLInterpreter.hpp"

namespace exatn {
std::string run_token_collector(clang::Preprocessor &PP,
                                clang::CachedTokens &Toks) {

  // Merge Toks into
  std::stringstream ss;
  for (int i = 0; i < Toks.size(); i++) {
    ss << PP.getSpelling(Toks[i]);
  }

  std::stringstream new_code_ss;
  parser::TAProLInterpreter interpreter;
  interpreter.interpret(ss.str(), new_code_ss);
  std::cout << new_code_ss.str() << "\n";

  return "new_code";
}

} // namespace exatn
