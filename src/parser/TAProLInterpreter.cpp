#include "TAProLInterpreter.hpp"
#include "TAProLLexer.h"
#include "TAProLListenerImpl.hpp"

using namespace antlr4;
using namespace taprol;

namespace exatn {

namespace parser {

void TAProLInterpreter::interpret(const std::string &src) {
  // Setup the Antlr Parser
  ANTLRInputStream input(src);
  TAProLLexer lexer(&input);
  lexer.removeErrorListeners();
  lexer.addErrorListener(new TAProLErrorListener());

  CommonTokenStream tokens(&lexer);
  TAProLParser parser(&tokens);
  parser.removeErrorListeners();
  parser.addErrorListener(new TAProLErrorListener());

  // Walk the Abstract Syntax Tree
  tree::ParseTree *tree = parser.taprolsrc();

  TAProLListenerImpl listener;
  tree::ParseTreeWalker::DEFAULT.walk(&listener, tree);
}

} // namespace parser
} // namespace exatn
