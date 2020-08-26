
// Generated from TAProL.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "TAProLListener.h"


namespace taprol {

/**
 * This class provides an empty implementation of TAProLListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  TAProLBaseListener : public TAProLListener {
public:

  virtual void enterTaprolsrc(TAProLParser::TaprolsrcContext * /*ctx*/) override { }
  virtual void exitTaprolsrc(TAProLParser::TaprolsrcContext * /*ctx*/) override { }

  virtual void enterEntry(TAProLParser::EntryContext * /*ctx*/) override { }
  virtual void exitEntry(TAProLParser::EntryContext * /*ctx*/) override { }

  virtual void enterScope(TAProLParser::ScopeContext * /*ctx*/) override { }
  virtual void exitScope(TAProLParser::ScopeContext * /*ctx*/) override { }

  virtual void enterScopename(TAProLParser::ScopenameContext * /*ctx*/) override { }
  virtual void exitScopename(TAProLParser::ScopenameContext * /*ctx*/) override { }

  virtual void enterGroupnamelist(TAProLParser::GroupnamelistContext * /*ctx*/) override { }
  virtual void exitGroupnamelist(TAProLParser::GroupnamelistContext * /*ctx*/) override { }

  virtual void enterGroupname(TAProLParser::GroupnameContext * /*ctx*/) override { }
  virtual void exitGroupname(TAProLParser::GroupnameContext * /*ctx*/) override { }

  virtual void enterCode(TAProLParser::CodeContext * /*ctx*/) override { }
  virtual void exitCode(TAProLParser::CodeContext * /*ctx*/) override { }

  virtual void enterLine(TAProLParser::LineContext * /*ctx*/) override { }
  virtual void exitLine(TAProLParser::LineContext * /*ctx*/) override { }

  virtual void enterStatement(TAProLParser::StatementContext * /*ctx*/) override { }
  virtual void exitStatement(TAProLParser::StatementContext * /*ctx*/) override { }

  virtual void enterSimpleop(TAProLParser::SimpleopContext * /*ctx*/) override { }
  virtual void exitSimpleop(TAProLParser::SimpleopContext * /*ctx*/) override { }

  virtual void enterCompositeop(TAProLParser::CompositeopContext * /*ctx*/) override { }
  virtual void exitCompositeop(TAProLParser::CompositeopContext * /*ctx*/) override { }

  virtual void enterSpace(TAProLParser::SpaceContext * /*ctx*/) override { }
  virtual void exitSpace(TAProLParser::SpaceContext * /*ctx*/) override { }

  virtual void enterNumfield(TAProLParser::NumfieldContext * /*ctx*/) override { }
  virtual void exitNumfield(TAProLParser::NumfieldContext * /*ctx*/) override { }

  virtual void enterSubspace(TAProLParser::SubspaceContext * /*ctx*/) override { }
  virtual void exitSubspace(TAProLParser::SubspaceContext * /*ctx*/) override { }

  virtual void enterSpacedeflist(TAProLParser::SpacedeflistContext * /*ctx*/) override { }
  virtual void exitSpacedeflist(TAProLParser::SpacedeflistContext * /*ctx*/) override { }

  virtual void enterSpacedef(TAProLParser::SpacedefContext * /*ctx*/) override { }
  virtual void exitSpacedef(TAProLParser::SpacedefContext * /*ctx*/) override { }

  virtual void enterSpacename(TAProLParser::SpacenameContext * /*ctx*/) override { }
  virtual void exitSpacename(TAProLParser::SpacenameContext * /*ctx*/) override { }

  virtual void enterRange(TAProLParser::RangeContext * /*ctx*/) override { }
  virtual void exitRange(TAProLParser::RangeContext * /*ctx*/) override { }

  virtual void enterLowerbound(TAProLParser::LowerboundContext * /*ctx*/) override { }
  virtual void exitLowerbound(TAProLParser::LowerboundContext * /*ctx*/) override { }

  virtual void enterUpperbound(TAProLParser::UpperboundContext * /*ctx*/) override { }
  virtual void exitUpperbound(TAProLParser::UpperboundContext * /*ctx*/) override { }

  virtual void enterIndex(TAProLParser::IndexContext * /*ctx*/) override { }
  virtual void exitIndex(TAProLParser::IndexContext * /*ctx*/) override { }

  virtual void enterIndexlist(TAProLParser::IndexlistContext * /*ctx*/) override { }
  virtual void exitIndexlist(TAProLParser::IndexlistContext * /*ctx*/) override { }

  virtual void enterIndexname(TAProLParser::IndexnameContext * /*ctx*/) override { }
  virtual void exitIndexname(TAProLParser::IndexnameContext * /*ctx*/) override { }

  virtual void enterAssign(TAProLParser::AssignContext * /*ctx*/) override { }
  virtual void exitAssign(TAProLParser::AssignContext * /*ctx*/) override { }

  virtual void enterDatacontainer(TAProLParser::DatacontainerContext * /*ctx*/) override { }
  virtual void exitDatacontainer(TAProLParser::DatacontainerContext * /*ctx*/) override { }

  virtual void enterMethodname(TAProLParser::MethodnameContext * /*ctx*/) override { }
  virtual void exitMethodname(TAProLParser::MethodnameContext * /*ctx*/) override { }

  virtual void enterRetrieve(TAProLParser::RetrieveContext * /*ctx*/) override { }
  virtual void exitRetrieve(TAProLParser::RetrieveContext * /*ctx*/) override { }

  virtual void enterLoad(TAProLParser::LoadContext * /*ctx*/) override { }
  virtual void exitLoad(TAProLParser::LoadContext * /*ctx*/) override { }

  virtual void enterSave(TAProLParser::SaveContext * /*ctx*/) override { }
  virtual void exitSave(TAProLParser::SaveContext * /*ctx*/) override { }

  virtual void enterTagname(TAProLParser::TagnameContext * /*ctx*/) override { }
  virtual void exitTagname(TAProLParser::TagnameContext * /*ctx*/) override { }

  virtual void enterDestroy(TAProLParser::DestroyContext * /*ctx*/) override { }
  virtual void exitDestroy(TAProLParser::DestroyContext * /*ctx*/) override { }

  virtual void enterTensorlist(TAProLParser::TensorlistContext * /*ctx*/) override { }
  virtual void exitTensorlist(TAProLParser::TensorlistContext * /*ctx*/) override { }

  virtual void enterNorm1(TAProLParser::Norm1Context * /*ctx*/) override { }
  virtual void exitNorm1(TAProLParser::Norm1Context * /*ctx*/) override { }

  virtual void enterNorm2(TAProLParser::Norm2Context * /*ctx*/) override { }
  virtual void exitNorm2(TAProLParser::Norm2Context * /*ctx*/) override { }

  virtual void enterMaxabs(TAProLParser::MaxabsContext * /*ctx*/) override { }
  virtual void exitMaxabs(TAProLParser::MaxabsContext * /*ctx*/) override { }

  virtual void enterScalar(TAProLParser::ScalarContext * /*ctx*/) override { }
  virtual void exitScalar(TAProLParser::ScalarContext * /*ctx*/) override { }

  virtual void enterScale(TAProLParser::ScaleContext * /*ctx*/) override { }
  virtual void exitScale(TAProLParser::ScaleContext * /*ctx*/) override { }

  virtual void enterPrefactor(TAProLParser::PrefactorContext * /*ctx*/) override { }
  virtual void exitPrefactor(TAProLParser::PrefactorContext * /*ctx*/) override { }

  virtual void enterCopy(TAProLParser::CopyContext * /*ctx*/) override { }
  virtual void exitCopy(TAProLParser::CopyContext * /*ctx*/) override { }

  virtual void enterAddition(TAProLParser::AdditionContext * /*ctx*/) override { }
  virtual void exitAddition(TAProLParser::AdditionContext * /*ctx*/) override { }

  virtual void enterContraction(TAProLParser::ContractionContext * /*ctx*/) override { }
  virtual void exitContraction(TAProLParser::ContractionContext * /*ctx*/) override { }

  virtual void enterCompositeproduct(TAProLParser::CompositeproductContext * /*ctx*/) override { }
  virtual void exitCompositeproduct(TAProLParser::CompositeproductContext * /*ctx*/) override { }

  virtual void enterTensornetwork(TAProLParser::TensornetworkContext * /*ctx*/) override { }
  virtual void exitTensornetwork(TAProLParser::TensornetworkContext * /*ctx*/) override { }

  virtual void enterTensor(TAProLParser::TensorContext * /*ctx*/) override { }
  virtual void exitTensor(TAProLParser::TensorContext * /*ctx*/) override { }

  virtual void enterConjtensor(TAProLParser::ConjtensorContext * /*ctx*/) override { }
  virtual void exitConjtensor(TAProLParser::ConjtensorContext * /*ctx*/) override { }

  virtual void enterTensorname(TAProLParser::TensornameContext * /*ctx*/) override { }
  virtual void exitTensorname(TAProLParser::TensornameContext * /*ctx*/) override { }

  virtual void enterId(TAProLParser::IdContext * /*ctx*/) override { }
  virtual void exitId(TAProLParser::IdContext * /*ctx*/) override { }

  virtual void enterComplex(TAProLParser::ComplexContext * /*ctx*/) override { }
  virtual void exitComplex(TAProLParser::ComplexContext * /*ctx*/) override { }

  virtual void enterReal(TAProLParser::RealContext * /*ctx*/) override { }
  virtual void exitReal(TAProLParser::RealContext * /*ctx*/) override { }

  virtual void enterString(TAProLParser::StringContext * /*ctx*/) override { }
  virtual void exitString(TAProLParser::StringContext * /*ctx*/) override { }

  virtual void enterComment(TAProLParser::CommentContext * /*ctx*/) override { }
  virtual void exitComment(TAProLParser::CommentContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

}  // namespace taprol
