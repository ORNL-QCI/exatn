
// Generated from TAProL.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"


namespace taprol {


class  TAProLParser : public antlr4::Parser {
public:
  enum {
    T__0 = 1, T__1 = 2, T__2 = 3, T__3 = 4, T__4 = 5, T__5 = 6, T__6 = 7, 
    T__7 = 8, T__8 = 9, T__9 = 10, T__10 = 11, T__11 = 12, T__12 = 13, T__13 = 14, 
    T__14 = 15, T__15 = 16, T__16 = 17, T__17 = 18, T__18 = 19, T__19 = 20, 
    T__20 = 21, T__21 = 22, T__22 = 23, T__23 = 24, T__24 = 25, T__25 = 26, 
    T__26 = 27, T__27 = 28, T__28 = 29, T__29 = 30, T__30 = 31, T__31 = 32, 
    T__32 = 33, COMMENT = 34, ID = 35, REAL = 36, FLOAT = 37, INT = 38, 
    ZINT = 39, STRING = 40, WS = 41, EOL = 42
  };

  enum {
    RuleTaprolsrc = 0, RuleEntry = 1, RuleScope = 2, RuleScopename = 3, 
    RuleGroupnamelist = 4, RuleGroupname = 5, RuleCode = 6, RuleLine = 7, 
    RuleStatement = 8, RuleSimpleop = 9, RuleCompositeop = 10, RuleSpace = 11, 
    RuleNumfield = 12, RuleSubspace = 13, RuleSpacedeflist = 14, RuleSpacedef = 15, 
    RuleSpacename = 16, RuleRange = 17, RuleLowerbound = 18, RuleUpperbound = 19, 
    RuleIndex = 20, RuleIndexlist = 21, RuleIndexname = 22, RuleAssign = 23, 
    RuleDatacontainer = 24, RuleMethodname = 25, RuleRetrieve = 26, RuleLoad = 27, 
    RuleSave = 28, RuleTagname = 29, RuleDestroy = 30, RuleTensorlist = 31, 
    RuleNorm = 32, RuleScalar = 33, RuleScale = 34, RulePrefactor = 35, 
    RuleCopy = 36, RuleAddition = 37, RuleContraction = 38, RuleCompositeproduct = 39, 
    RuleTensornetwork = 40, RuleTensor = 41, RuleConjtensor = 42, RuleTensorname = 43, 
    RuleId = 44, RuleComplex = 45, RuleReal = 46, RuleString = 47, RuleComment = 48
  };

  TAProLParser(antlr4::TokenStream *input);
  ~TAProLParser();

  virtual std::string getGrammarFileName() const override;
  virtual const antlr4::atn::ATN& getATN() const override { return _atn; };
  virtual const std::vector<std::string>& getTokenNames() const override { return _tokenNames; }; // deprecated: use vocabulary instead.
  virtual const std::vector<std::string>& getRuleNames() const override;
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;


  class TaprolsrcContext;
  class EntryContext;
  class ScopeContext;
  class ScopenameContext;
  class GroupnamelistContext;
  class GroupnameContext;
  class CodeContext;
  class LineContext;
  class StatementContext;
  class SimpleopContext;
  class CompositeopContext;
  class SpaceContext;
  class NumfieldContext;
  class SubspaceContext;
  class SpacedeflistContext;
  class SpacedefContext;
  class SpacenameContext;
  class RangeContext;
  class LowerboundContext;
  class UpperboundContext;
  class IndexContext;
  class IndexlistContext;
  class IndexnameContext;
  class AssignContext;
  class DatacontainerContext;
  class MethodnameContext;
  class RetrieveContext;
  class LoadContext;
  class SaveContext;
  class TagnameContext;
  class DestroyContext;
  class TensorlistContext;
  class NormContext;
  class ScalarContext;
  class ScaleContext;
  class PrefactorContext;
  class CopyContext;
  class AdditionContext;
  class ContractionContext;
  class CompositeproductContext;
  class TensornetworkContext;
  class TensorContext;
  class ConjtensorContext;
  class TensornameContext;
  class IdContext;
  class ComplexContext;
  class RealContext;
  class StringContext;
  class CommentContext; 

  class  TaprolsrcContext : public antlr4::ParserRuleContext {
  public:
    TaprolsrcContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    EntryContext *entry();
    std::vector<ScopeContext *> scope();
    ScopeContext* scope(size_t i);
    CodeContext *code();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  TaprolsrcContext* taprolsrc();

  class  EntryContext : public antlr4::ParserRuleContext {
  public:
    EntryContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ScopenameContext *scopename();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  EntryContext* entry();

  class  ScopeContext : public antlr4::ParserRuleContext {
  public:
    ScopeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ScopenameContext *> scopename();
    ScopenameContext* scopename(size_t i);
    CodeContext *code();
    GroupnamelistContext *groupnamelist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ScopeContext* scope();

  class  ScopenameContext : public antlr4::ParserRuleContext {
  public:
    ScopenameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ScopenameContext* scopename();

  class  GroupnamelistContext : public antlr4::ParserRuleContext {
  public:
    GroupnamelistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<GroupnameContext *> groupname();
    GroupnameContext* groupname(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  GroupnamelistContext* groupnamelist();

  class  GroupnameContext : public antlr4::ParserRuleContext {
  public:
    GroupnameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  GroupnameContext* groupname();

  class  CodeContext : public antlr4::ParserRuleContext {
  public:
    CodeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<LineContext *> line();
    LineContext* line(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  CodeContext* code();

  class  LineContext : public antlr4::ParserRuleContext {
  public:
    LineContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    StatementContext *statement();
    CommentContext *comment();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  LineContext* line();

  class  StatementContext : public antlr4::ParserRuleContext {
  public:
    StatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SpaceContext *space();
    SubspaceContext *subspace();
    IndexContext *index();
    SimpleopContext *simpleop();
    CompositeopContext *compositeop();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  StatementContext* statement();

  class  SimpleopContext : public antlr4::ParserRuleContext {
  public:
    SimpleopContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    AssignContext *assign();
    RetrieveContext *retrieve();
    LoadContext *load();
    SaveContext *save();
    DestroyContext *destroy();
    NormContext *norm();
    ScaleContext *scale();
    CopyContext *copy();
    AdditionContext *addition();
    ContractionContext *contraction();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SimpleopContext* simpleop();

  class  CompositeopContext : public antlr4::ParserRuleContext {
  public:
    CompositeopContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    CompositeproductContext *compositeproduct();
    TensornetworkContext *tensornetwork();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  CompositeopContext* compositeop();

  class  SpaceContext : public antlr4::ParserRuleContext {
  public:
    SpaceContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    NumfieldContext *numfield();
    SpacedeflistContext *spacedeflist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SpaceContext* space();

  class  NumfieldContext : public antlr4::ParserRuleContext {
  public:
    NumfieldContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  NumfieldContext* numfield();

  class  SubspaceContext : public antlr4::ParserRuleContext {
  public:
    SubspaceContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SpacedeflistContext *spacedeflist();
    SpacenameContext *spacename();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SubspaceContext* subspace();

  class  SpacedeflistContext : public antlr4::ParserRuleContext {
  public:
    SpacedeflistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<SpacedefContext *> spacedef();
    SpacedefContext* spacedef(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SpacedeflistContext* spacedeflist();

  class  SpacedefContext : public antlr4::ParserRuleContext {
  public:
    SpacedefContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SpacenameContext *spacename();
    RangeContext *range();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SpacedefContext* spacedef();

  class  SpacenameContext : public antlr4::ParserRuleContext {
  public:
    SpacenameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SpacenameContext* spacename();

  class  RangeContext : public antlr4::ParserRuleContext {
  public:
    RangeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    LowerboundContext *lowerbound();
    UpperboundContext *upperbound();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  RangeContext* range();

  class  LowerboundContext : public antlr4::ParserRuleContext {
  public:
    LowerboundContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *INT();
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  LowerboundContext* lowerbound();

  class  UpperboundContext : public antlr4::ParserRuleContext {
  public:
    UpperboundContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *INT();
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  UpperboundContext* upperbound();

  class  IndexContext : public antlr4::ParserRuleContext {
  public:
    IndexContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SpacenameContext *spacename();
    IndexlistContext *indexlist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  IndexContext* index();

  class  IndexlistContext : public antlr4::ParserRuleContext {
  public:
    IndexlistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<IndexnameContext *> indexname();
    IndexnameContext* indexname(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  IndexlistContext* indexlist();

  class  IndexnameContext : public antlr4::ParserRuleContext {
  public:
    IndexnameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  IndexnameContext* indexname();

  class  AssignContext : public antlr4::ParserRuleContext {
  public:
    AssignContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TensorContext *tensor();
    RealContext *real();
    ComplexContext *complex();
    DatacontainerContext *datacontainer();
    MethodnameContext *methodname();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  AssignContext* assign();

  class  DatacontainerContext : public antlr4::ParserRuleContext {
  public:
    DatacontainerContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  DatacontainerContext* datacontainer();

  class  MethodnameContext : public antlr4::ParserRuleContext {
  public:
    MethodnameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    StringContext *string();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  MethodnameContext* methodname();

  class  RetrieveContext : public antlr4::ParserRuleContext {
  public:
    RetrieveContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    DatacontainerContext *datacontainer();
    TensornameContext *tensorname();
    TensorContext *tensor();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  RetrieveContext* retrieve();

  class  LoadContext : public antlr4::ParserRuleContext {
  public:
    LoadContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TagnameContext *tagname();
    TensorContext *tensor();
    TensornameContext *tensorname();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  LoadContext* load();

  class  SaveContext : public antlr4::ParserRuleContext {
  public:
    SaveContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TagnameContext *tagname();
    TensorContext *tensor();
    TensornameContext *tensorname();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SaveContext* save();

  class  TagnameContext : public antlr4::ParserRuleContext {
  public:
    TagnameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    StringContext *string();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  TagnameContext* tagname();

  class  DestroyContext : public antlr4::ParserRuleContext {
  public:
    DestroyContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TensornameContext *tensorname();
    TensorContext *tensor();
    TensorlistContext *tensorlist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  DestroyContext* destroy();

  class  TensorlistContext : public antlr4::ParserRuleContext {
  public:
    TensorlistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TensornameContext *> tensorname();
    TensornameContext* tensorname(size_t i);
    std::vector<TensorContext *> tensor();
    TensorContext* tensor(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  TensorlistContext* tensorlist();

  class  NormContext : public antlr4::ParserRuleContext {
  public:
    NormContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ScalarContext *scalar();
    TensornameContext *tensorname();
    TensorContext *tensor();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  NormContext* norm();

  class  ScalarContext : public antlr4::ParserRuleContext {
  public:
    ScalarContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ScalarContext* scalar();

  class  ScaleContext : public antlr4::ParserRuleContext {
  public:
    ScaleContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TensorContext *tensor();
    PrefactorContext *prefactor();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ScaleContext* scale();

  class  PrefactorContext : public antlr4::ParserRuleContext {
  public:
    PrefactorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    RealContext *real();
    ComplexContext *complex();
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PrefactorContext* prefactor();

  class  CopyContext : public antlr4::ParserRuleContext {
  public:
    CopyContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TensorContext *> tensor();
    TensorContext* tensor(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  CopyContext* copy();

  class  AdditionContext : public antlr4::ParserRuleContext {
  public:
    AdditionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TensorContext *> tensor();
    TensorContext* tensor(size_t i);
    ConjtensorContext *conjtensor();
    PrefactorContext *prefactor();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  AdditionContext* addition();

  class  ContractionContext : public antlr4::ParserRuleContext {
  public:
    ContractionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TensorContext *> tensor();
    TensorContext* tensor(size_t i);
    std::vector<ConjtensorContext *> conjtensor();
    ConjtensorContext* conjtensor(size_t i);
    PrefactorContext *prefactor();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ContractionContext* contraction();

  class  CompositeproductContext : public antlr4::ParserRuleContext {
  public:
    CompositeproductContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TensorContext *> tensor();
    TensorContext* tensor(size_t i);
    std::vector<ConjtensorContext *> conjtensor();
    ConjtensorContext* conjtensor(size_t i);
    PrefactorContext *prefactor();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  CompositeproductContext* compositeproduct();

  class  TensornetworkContext : public antlr4::ParserRuleContext {
  public:
    TensornetworkContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TensorContext *> tensor();
    TensorContext* tensor(size_t i);
    std::vector<ConjtensorContext *> conjtensor();
    ConjtensorContext* conjtensor(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  TensornetworkContext* tensornetwork();

  class  TensorContext : public antlr4::ParserRuleContext {
  public:
    TensorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TensornameContext *tensorname();
    IndexlistContext *indexlist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  TensorContext* tensor();

  class  ConjtensorContext : public antlr4::ParserRuleContext {
  public:
    ConjtensorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TensornameContext *tensorname();
    IndexlistContext *indexlist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ConjtensorContext* conjtensor();

  class  TensornameContext : public antlr4::ParserRuleContext {
  public:
    TensornameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  TensornameContext* tensorname();

  class  IdContext : public antlr4::ParserRuleContext {
  public:
    IdContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  IdContext* id();

  class  ComplexContext : public antlr4::ParserRuleContext {
  public:
    ComplexContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<RealContext *> real();
    RealContext* real(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ComplexContext* complex();

  class  RealContext : public antlr4::ParserRuleContext {
  public:
    RealContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *REAL();
    antlr4::tree::TerminalNode *FLOAT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  RealContext* real();

  class  StringContext : public antlr4::ParserRuleContext {
  public:
    StringContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *STRING();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  StringContext* string();

  class  CommentContext : public antlr4::ParserRuleContext {
  public:
    CommentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *COMMENT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  CommentContext* comment();


private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

}  // namespace taprol
