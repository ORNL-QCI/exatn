
// Generated from TAProL.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "TAProLParser.h"


namespace taprol {

/**
 * This interface defines an abstract listener for a parse tree produced by TAProLParser.
 */
class  TAProLListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterTaprolsrc(TAProLParser::TaprolsrcContext *ctx) = 0;
  virtual void exitTaprolsrc(TAProLParser::TaprolsrcContext *ctx) = 0;

  virtual void enterEntry(TAProLParser::EntryContext *ctx) = 0;
  virtual void exitEntry(TAProLParser::EntryContext *ctx) = 0;

  virtual void enterScope(TAProLParser::ScopeContext *ctx) = 0;
  virtual void exitScope(TAProLParser::ScopeContext *ctx) = 0;

  virtual void enterGroupnamelist(TAProLParser::GroupnamelistContext *ctx) = 0;
  virtual void exitGroupnamelist(TAProLParser::GroupnamelistContext *ctx) = 0;

  virtual void enterCode(TAProLParser::CodeContext *ctx) = 0;
  virtual void exitCode(TAProLParser::CodeContext *ctx) = 0;

  virtual void enterLine(TAProLParser::LineContext *ctx) = 0;
  virtual void exitLine(TAProLParser::LineContext *ctx) = 0;

  virtual void enterStatement(TAProLParser::StatementContext *ctx) = 0;
  virtual void exitStatement(TAProLParser::StatementContext *ctx) = 0;

  virtual void enterCompositeop(TAProLParser::CompositeopContext *ctx) = 0;
  virtual void exitCompositeop(TAProLParser::CompositeopContext *ctx) = 0;

  virtual void enterSimpleop(TAProLParser::SimpleopContext *ctx) = 0;
  virtual void exitSimpleop(TAProLParser::SimpleopContext *ctx) = 0;

  virtual void enterIndex(TAProLParser::IndexContext *ctx) = 0;
  virtual void exitIndex(TAProLParser::IndexContext *ctx) = 0;

  virtual void enterIndexlist(TAProLParser::IndexlistContext *ctx) = 0;
  virtual void exitIndexlist(TAProLParser::IndexlistContext *ctx) = 0;

  virtual void enterIndexname(TAProLParser::IndexnameContext *ctx) = 0;
  virtual void exitIndexname(TAProLParser::IndexnameContext *ctx) = 0;

  virtual void enterSubspace(TAProLParser::SubspaceContext *ctx) = 0;
  virtual void exitSubspace(TAProLParser::SubspaceContext *ctx) = 0;

  virtual void enterSpace(TAProLParser::SpaceContext *ctx) = 0;
  virtual void exitSpace(TAProLParser::SpaceContext *ctx) = 0;

  virtual void enterSpacename(TAProLParser::SpacenameContext *ctx) = 0;
  virtual void exitSpacename(TAProLParser::SpacenameContext *ctx) = 0;

  virtual void enterNumfield(TAProLParser::NumfieldContext *ctx) = 0;
  virtual void exitNumfield(TAProLParser::NumfieldContext *ctx) = 0;

  virtual void enterSpacedeflist(TAProLParser::SpacedeflistContext *ctx) = 0;
  virtual void exitSpacedeflist(TAProLParser::SpacedeflistContext *ctx) = 0;

  virtual void enterSpacedef(TAProLParser::SpacedefContext *ctx) = 0;
  virtual void exitSpacedef(TAProLParser::SpacedefContext *ctx) = 0;

  virtual void enterRange(TAProLParser::RangeContext *ctx) = 0;
  virtual void exitRange(TAProLParser::RangeContext *ctx) = 0;

  virtual void enterLowerbound(TAProLParser::LowerboundContext *ctx) = 0;
  virtual void exitLowerbound(TAProLParser::LowerboundContext *ctx) = 0;

  virtual void enterUpperbound(TAProLParser::UpperboundContext *ctx) = 0;
  virtual void exitUpperbound(TAProLParser::UpperboundContext *ctx) = 0;

  virtual void enterAssignment(TAProLParser::AssignmentContext *ctx) = 0;
  virtual void exitAssignment(TAProLParser::AssignmentContext *ctx) = 0;

  virtual void enterMethodname(TAProLParser::MethodnameContext *ctx) = 0;
  virtual void exitMethodname(TAProLParser::MethodnameContext *ctx) = 0;

  virtual void enterLoad(TAProLParser::LoadContext *ctx) = 0;
  virtual void exitLoad(TAProLParser::LoadContext *ctx) = 0;

  virtual void enterSave(TAProLParser::SaveContext *ctx) = 0;
  virtual void exitSave(TAProLParser::SaveContext *ctx) = 0;

  virtual void enterTagname(TAProLParser::TagnameContext *ctx) = 0;
  virtual void exitTagname(TAProLParser::TagnameContext *ctx) = 0;

  virtual void enterDestroy(TAProLParser::DestroyContext *ctx) = 0;
  virtual void exitDestroy(TAProLParser::DestroyContext *ctx) = 0;

  virtual void enterTensorlist(TAProLParser::TensorlistContext *ctx) = 0;
  virtual void exitTensorlist(TAProLParser::TensorlistContext *ctx) = 0;

  virtual void enterCopy(TAProLParser::CopyContext *ctx) = 0;
  virtual void exitCopy(TAProLParser::CopyContext *ctx) = 0;

  virtual void enterScale(TAProLParser::ScaleContext *ctx) = 0;
  virtual void exitScale(TAProLParser::ScaleContext *ctx) = 0;

  virtual void enterUnaryop(TAProLParser::UnaryopContext *ctx) = 0;
  virtual void exitUnaryop(TAProLParser::UnaryopContext *ctx) = 0;

  virtual void enterBinaryop(TAProLParser::BinaryopContext *ctx) = 0;
  virtual void exitBinaryop(TAProLParser::BinaryopContext *ctx) = 0;

  virtual void enterPrefactor(TAProLParser::PrefactorContext *ctx) = 0;
  virtual void exitPrefactor(TAProLParser::PrefactorContext *ctx) = 0;

  virtual void enterCompositeproduct(TAProLParser::CompositeproductContext *ctx) = 0;
  virtual void exitCompositeproduct(TAProLParser::CompositeproductContext *ctx) = 0;

  virtual void enterTensornetwork(TAProLParser::TensornetworkContext *ctx) = 0;
  virtual void exitTensornetwork(TAProLParser::TensornetworkContext *ctx) = 0;

  virtual void enterConjtensor(TAProLParser::ConjtensorContext *ctx) = 0;
  virtual void exitConjtensor(TAProLParser::ConjtensorContext *ctx) = 0;

  virtual void enterTensor(TAProLParser::TensorContext *ctx) = 0;
  virtual void exitTensor(TAProLParser::TensorContext *ctx) = 0;

  virtual void enterTensorname(TAProLParser::TensornameContext *ctx) = 0;
  virtual void exitTensorname(TAProLParser::TensornameContext *ctx) = 0;

  virtual void enterId(TAProLParser::IdContext *ctx) = 0;
  virtual void exitId(TAProLParser::IdContext *ctx) = 0;

  virtual void enterComplex(TAProLParser::ComplexContext *ctx) = 0;
  virtual void exitComplex(TAProLParser::ComplexContext *ctx) = 0;

  virtual void enterReal(TAProLParser::RealContext *ctx) = 0;
  virtual void exitReal(TAProLParser::RealContext *ctx) = 0;

  virtual void enterString(TAProLParser::StringContext *ctx) = 0;
  virtual void exitString(TAProLParser::StringContext *ctx) = 0;

  virtual void enterComment(TAProLParser::CommentContext *ctx) = 0;
  virtual void exitComment(TAProLParser::CommentContext *ctx) = 0;


};

}  // namespace taprol
