#ifndef EXATN_TAPROLLISTENER_H
#define EXATN_TAPROLLISTENER_H

#include <TAProLBaseListener.h>

using namespace taprol;

namespace exatn {
namespace parser {

/**
 */
class TAProLListener : public TAProLBaseListener {
public:

  TAProLListener();

  void enterEntry(TAProLParser::EntryContext * ctx) override;
  void exitEntry(TAProLParser::EntryContext * ctx) override;

  void enterScope(TAProLParser::ScopeContext * ctx) override;
  void exitScope(TAProLParser::ScopeContext * ctx) override;

  void enterSimpleop(TAProLParser::SimpleopContext * ctx) override;
  void exitSimpleop(TAProLParser::SimpleopContext * ctx) override;

  void enterCompositeop(TAProLParser::CompositeopContext * ctx) override;
  void exitCompositeop(TAProLParser::CompositeopContext * ctx) override;

  void enterSpace(TAProLParser::SpaceContext * ctx) override;
  void exitSpace(TAProLParser::SpaceContext * ctx) override;

  void enterSubspace(TAProLParser::SubspaceContext * ctx) override;
  void exitSubspace(TAProLParser::SubspaceContext * ctx) override;

  void enterIndex(TAProLParser::IndexContext * ctx) override;
  void exitIndex(TAProLParser::IndexContext * ctx) override;

  void enterAssignment(TAProLParser::AssignmentContext * ctx) override;
  void exitAssignment(TAProLParser::AssignmentContext * ctx) override;

  void enterLoad(TAProLParser::LoadContext * ctx) override;
  void exitLoad(TAProLParser::LoadContext * ctx) override;

  void enterSave(TAProLParser::SaveContext * ctx) override;
  void exitSave(TAProLParser::SaveContext * ctx) override;

  void enterDestroy(TAProLParser::DestroyContext * ctx) override;
  void exitDestroy(TAProLParser::DestroyContext * ctx) override;

  void enterCopy(TAProLParser::CopyContext * ctx) override;
  void exitCopy(TAProLParser::CopyContext * ctx) override;

  void enterScale(TAProLParser::ScaleContext * ctx) override;
  void exitScale(TAProLParser::ScaleContext * ctx) override;

  void enterUnaryop(TAProLParser::UnaryopContext * ctx) override;
  void exitUnaryop(TAProLParser::UnaryopContext * ctx) override;

  void enterBinaryop(TAProLParser::BinaryopContext * ctx) override;
  void exitBinaryop(TAProLParser::BinaryopContext * ctx) override;

  void enterCompositeproduct(TAProLParser::CompositeproductContext * ctx) override;
  void exitCompositeproduct(TAProLParser::CompositeproductContext * ctx) override;

  void enterTensornetwork(TAProLParser::TensornetworkContext * ctx) override;
  void exitTensornetwork(TAProLParser::TensornetworkContext * ctx) override;

};

}

}

#endif
