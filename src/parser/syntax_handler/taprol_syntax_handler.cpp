#include <iostream>
#include <regex>
#include <sstream>

#include "taprol_parser_util.hpp"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

namespace {

class TaProlSyntaxHandler : public SyntaxHandler {
public:
  TaProlSyntaxHandler() : SyntaxHandler("taprol"){}
  
  void GetReplacement(Preprocessor &PP, Declarator &D, CachedTokens &Toks,
                      llvm::raw_string_ostream &OS) {
    const DeclaratorChunk::FunctionTypeInfo &FTI = D.getFunctionTypeInfo();
    auto kernel_name = D.getName().Identifier->getName().str();

    std::vector<std::string> program_parameters, program_arg_types;
    std::vector<std::string> bufferNames;
    for (unsigned int ii = 0; ii < FTI.NumParams; ii++) {
      auto &paramInfo = FTI.Params[ii];
      Token IdentToken, TypeToken;
      auto ident = paramInfo.Ident;
      auto &decl = paramInfo.Param;
      auto parm_var_decl = cast<ParmVarDecl>(decl);
      PP.getRawToken(paramInfo.IdentLoc, IdentToken);
      PP.getRawToken(decl->getBeginLoc(), TypeToken);

      auto type = QualType::getAsString(parm_var_decl->getType().split(),
                                        PrintingPolicy{{}});
      auto var = PP.getSpelling(IdentToken);

      if (type == "class xacc::internal_compiler::qreg") {
        bufferNames.push_back(ident->getName().str());
        type = "qreg";
      } else if (type == "qcor::qreg") {
        bufferNames.push_back(ident->getName().str());
        type = "qreg";
      }

      program_arg_types.push_back(type);
      program_parameters.push_back(var);
    }

    // Collect the tokens into a string
    for (auto &tok : Toks) {
      std::cout << "Tok: " << PP.getSpelling(tok) << "\n";
    }

    auto new_code = exatn::run_token_collector(PP, Toks);

    auto s = OS.str();
    std::cout << "[taprol syntax-handler] Rewriting " << kernel_name
              << " to\n\n"
              << s;
  }

  void AddToPredefines(llvm::raw_string_ostream &OS) {
    // OS << "#include \"exatn.hpp\"\n";
  }
};
} // namespace
static SyntaxHandlerRegistry::Add<TaProlSyntaxHandler>
    X("taprol", "Taprol syntax handler");
