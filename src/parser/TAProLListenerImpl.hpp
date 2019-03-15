#ifndef EXATN_TAPROLLISTENERIMPL_H
#define EXATN_TAPROLLISTENERIMPL_H

#include "TAProLParser.h"

#include "TAProLBaseListener.h"

using namespace taprol;

namespace exatn {
namespace parser {

/**
 */
class TAProLListenerImpl : public TAProLBaseListener {

protected:
  std::string entryValue = "";

public:
  void enterEntry(TAProLParser::EntryContext *ctx) override;

  void enterScope(TAProLParser::ScopeContext *ctx) override;

  void enterSpace(TAProLParser::SpaceContext *ctx) override;

  void enterSubspace(TAProLParser::SubspaceContext *ctx) override;

  void enterIndex(TAProLParser::IndexContext *ctx) override;

  void enterAssignment(TAProLParser::AssignmentContext *ctx) override;

  void enterLoad(TAProLParser::LoadContext *ctx) override;

  void enterSave(TAProLParser::SaveContext *ctx) override;

  void enterDestroy(TAProLParser::DestroyContext *ctx) override;

  void enterCopy(TAProLParser::CopyContext *ctx) override;

  void enterScale(TAProLParser::ScaleContext *ctx) override;

  void enterUnaryop(TAProLParser::UnaryopContext *ctx) override;

  void enterBinaryop(TAProLParser::BinaryopContext *ctx) override;

  void
  enterCompositeproduct(TAProLParser::CompositeproductContext *ctx) override;

  void enterTensornetwork(TAProLParser::TensornetworkContext *ctx) override;
};

} // namespace parser

} // namespace exatn

#endif
