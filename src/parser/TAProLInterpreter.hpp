#ifndef PARSER_TAPROLPARSER_HPP_
#define PARSER_TAPROLPARSER_HPP_
#include "antlr4-runtime.h"
#include "num_server.hpp"

namespace exatn {

namespace parser {

/**
 * An Antlr error listener for handling parsing errors.
 */
class TAProLErrorListener : public antlr4::BaseErrorListener {
public:
  void syntaxError(antlr4::Recognizer *recognizer,
                   antlr4::Token *offendingSymbol, size_t line,
                   size_t charPositionInLine, const std::string &msg,
                   std::exception_ptr e) override {
    std::ostringstream output;
    output << "Invalid TAProL source: ";
    output << "line " << line << ":" << charPositionInLine << " " << msg;
    std::cerr << output.str() << "\n";
  }
};

/**
 */
class TAProLInterpreter {
public:
  std::shared_ptr<exatn::numerics::NumServer> numerics_server;

  TAProLInterpreter() : numerics_server(std::make_shared<exatn::numerics::NumServer>()) {}
// NumServer(const NumServer &) = delete;
//  NumServer & operator=(const NumServer &) = delete;
//  NumServer(NumServer &&) noexcept = default;
//  NumServer & operator=(NumServer &&) noexcept = default;
//  ~NumServer() = default;
  void interpret(const std::string &src);

  virtual ~TAProLInterpreter() {}
};

} // namespace parser
} // namespace exatn

#endif