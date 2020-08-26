#ifndef EXATN_TAPROLLISTENERCPPIMPL_HPP_
#define EXATN_TAPROLLISTENERCPPIMPL_HPP_

#include "TAProLParser.h"
#include "TAProLBaseListener.h"

#include <iostream>
#include <sstream>
#include <string>
#include <map>

using namespace taprol;

namespace exatn {

namespace parser {

class TAProLListenerCPPImpl : public TAProLBaseListener {

public:

  virtual void enterEntry(TAProLParser::EntryContext * ctx) override;

  virtual void enterScope(TAProLParser::ScopeContext * ctx) override;
  virtual void exitScope(TAProLParser::ScopeContext * ctx) override;

  virtual void enterCode(TAProLParser::CodeContext * ctx) override;
  virtual void exitCode(TAProLParser::CodeContext * ctx) override;

  virtual void enterSpace(TAProLParser::SpaceContext * ctx) override;

  virtual void enterSubspace(TAProLParser::SubspaceContext * ctx) override;

  virtual void enterIndex(TAProLParser::IndexContext * ctx) override;

  virtual void enterAssign(TAProLParser::AssignContext * ctx) override;

  virtual void enterRetrieve(TAProLParser::RetrieveContext * ctx) override;

  virtual void enterLoad(TAProLParser::LoadContext * ctx) override;

  virtual void enterSave(TAProLParser::SaveContext * ctx) override;

  virtual void enterDestroy(TAProLParser::DestroyContext * ctx) override;

  virtual void enterNorm1(TAProLParser::Norm1Context * ctx) override;

  virtual void enterNorm2(TAProLParser::Norm2Context * ctx) override;

  virtual void enterMaxabs(TAProLParser::MaxabsContext * ctx) override;

  virtual void enterScale(TAProLParser::ScaleContext * ctx) override;

  virtual void enterCopy(TAProLParser::CopyContext * ctx) override;

  virtual void enterAddition(TAProLParser::AdditionContext * ctx) override;

  virtual void enterContraction(TAProLParser::ContractionContext * ctx) override;

  virtual void enterCompositeproduct(TAProLParser::CompositeproductContext * ctx) override;

  virtual void enterTensornetwork(TAProLParser::TensornetworkContext * ctx) override;

  virtual ~TAProLListenerCPPImpl() {
    std::cout << "#DEBUG(exatn::parser::cpp): C++ source generated from TAProL source:" << std::endl;
    std::cout << cpp_source.str();
  }

protected:

  std::ostringstream cpp_source;          //output C++ code buffer
  std::map<std::string,std::string> args; //function arg name --> function arg type

};

} // namespace parser

} // namespace exatn

#endif //EXATN_TAPROLLISTENERCPPIMPL_HPP_
