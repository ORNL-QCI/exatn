
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
    T__26 = 27, T__27 = 28, T__28 = 29, T__29 = 30, COMMENT = 31, ID = 32, 
    REAL = 33, INT = 34, STRING = 35, WS = 36, EOL = 37
  };

  enum {
    RuleTaprolsrc = 0, RuleEntry = 1, RuleScope = 2, RuleCode = 3, RuleLine = 4, 
    RuleStatement = 5, RuleSimpleop = 6, RuleCompositeop = 7, RuleSpace = 8, 
    RuleSubspace = 9, RuleSpacelist = 10, RuleIndex = 11, RuleIdx = 12, 
    RuleAssignment = 13, RuleLoad = 14, RuleSave = 15, RuleDestroy = 16, 
    RuleCopy = 17, RuleScale = 18, RuleUnaryop = 19, RuleBinaryop = 20, 
    RuleCompositeproduct = 21, RuleTensornetwork = 22, RuleTensorname = 23, 
    RuleTensor = 24, RuleConjtensor = 25, RuleActualindex = 26, RuleIndexlist = 27, 
    RuleComment = 28, RuleRange = 29, RuleGroupnamelist = 30, RuleGroupname = 31, 
    RuleId = 32, RuleComplex = 33, RuleReal = 34, RuleString = 35
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
  class CodeContext;
  class LineContext;
  class StatementContext;
  class SimpleopContext;
  class CompositeopContext;
  class SpaceContext;
  class SubspaceContext;
  class SpacelistContext;
  class IndexContext;
  class IdxContext;
  class AssignmentContext;
  class LoadContext;
  class SaveContext;
  class DestroyContext;
  class CopyContext;
  class ScaleContext;
  class UnaryopContext;
  class BinaryopContext;
  class CompositeproductContext;
  class TensornetworkContext;
  class TensornameContext;
  class TensorContext;
  class ConjtensorContext;
  class ActualindexContext;
  class IndexlistContext;
  class CommentContext;
  class RangeContext;
  class GroupnamelistContext;
  class GroupnameContext;
  class IdContext;
  class ComplexContext;
  class RealContext;
  class StringContext; 

  class  TaprolsrcContext : public antlr4::ParserRuleContext {
  public:
    TaprolsrcContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    EntryContext *entry();
    std::vector<ScopeContext *> scope();
    ScopeContext* scope(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  TaprolsrcContext* taprolsrc();

  class  EntryContext : public antlr4::ParserRuleContext {
  public:
    antlr4::Token *entryName = nullptr;;
    EntryContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  EntryContext* entry();

  class  ScopeContext : public antlr4::ParserRuleContext {
  public:
    TAProLParser::IdContext *scopebeginname = nullptr;;
    TAProLParser::IdContext *scopeendname = nullptr;;
    ScopeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    CodeContext *code();
    std::vector<IdContext *> id();
    IdContext* id(size_t i);
    GroupnamelistContext *groupnamelist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ScopeContext* scope();

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
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);
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
    AssignmentContext *assignment();
    LoadContext *load();
    SaveContext *save();
    DestroyContext *destroy();
    CopyContext *copy();
    ScaleContext *scale();
    UnaryopContext *unaryop();
    BinaryopContext *binaryop();

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
    SpacelistContext *spacelist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SpaceContext* space();

  class  SubspaceContext : public antlr4::ParserRuleContext {
  public:
    TAProLParser::IdContext *spacename = nullptr;;
    SubspaceContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SpacelistContext *spacelist();
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SubspaceContext* subspace();

  class  SpacelistContext : public antlr4::ParserRuleContext {
  public:
    TAProLParser::IdContext *spacename = nullptr;;
    SpacelistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    RangeContext *range();
    IdContext *id();
    std::vector<SpacelistContext *> spacelist();
    SpacelistContext* spacelist(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SpacelistContext* spacelist();

  class  IndexContext : public antlr4::ParserRuleContext {
  public:
    TAProLParser::IdContext *subspacename = nullptr;;
    IndexContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<IdxContext *> idx();
    IdxContext* idx(size_t i);
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  IndexContext* index();

  class  IdxContext : public antlr4::ParserRuleContext {
  public:
    IdxContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  IdxContext* idx();

  class  AssignmentContext : public antlr4::ParserRuleContext {
  public:
    AssignmentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TensorContext *tensor();
    StringContext *string();
    ComplexContext *complex();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  AssignmentContext* assignment();

  class  LoadContext : public antlr4::ParserRuleContext {
  public:
    LoadContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TensorContext *tensor();
    StringContext *string();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  LoadContext* load();

  class  SaveContext : public antlr4::ParserRuleContext {
  public:
    SaveContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TensorContext *tensor();
    StringContext *string();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SaveContext* save();

  class  DestroyContext : public antlr4::ParserRuleContext {
  public:
    DestroyContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TensornameContext *> tensorname();
    TensornameContext* tensorname(size_t i);
    std::vector<TensorContext *> tensor();
    TensorContext* tensor(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  DestroyContext* destroy();

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

  class  ScaleContext : public antlr4::ParserRuleContext {
  public:
    ScaleContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TensorContext *tensor();
    RealContext *real();
    ComplexContext *complex();
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ScaleContext* scale();

  class  UnaryopContext : public antlr4::ParserRuleContext {
  public:
    UnaryopContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TensorContext *> tensor();
    TensorContext* tensor(size_t i);
    ConjtensorContext *conjtensor();
    RealContext *real();
    ComplexContext *complex();
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  UnaryopContext* unaryop();

  class  BinaryopContext : public antlr4::ParserRuleContext {
  public:
    BinaryopContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TensorContext *> tensor();
    TensorContext* tensor(size_t i);
    std::vector<ConjtensorContext *> conjtensor();
    ConjtensorContext* conjtensor(size_t i);
    RealContext *real();
    ComplexContext *complex();
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  BinaryopContext* binaryop();

  class  CompositeproductContext : public antlr4::ParserRuleContext {
  public:
    CompositeproductContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TensorContext *> tensor();
    TensorContext* tensor(size_t i);
    std::vector<ConjtensorContext *> conjtensor();
    ConjtensorContext* conjtensor(size_t i);
    RealContext *real();
    ComplexContext *complex();
    IdContext *id();

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

  class  TensornameContext : public antlr4::ParserRuleContext {
  public:
    TensornameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  TensornameContext* tensorname();

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

  class  ActualindexContext : public antlr4::ParserRuleContext {
  public:
    ActualindexContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();
    antlr4::tree::TerminalNode *INT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ActualindexContext* actualindex();

  class  IndexlistContext : public antlr4::ParserRuleContext {
  public:
    IndexlistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ActualindexContext *> actualindex();
    ActualindexContext* actualindex(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  IndexlistContext* indexlist();

  class  CommentContext : public antlr4::ParserRuleContext {
  public:
    CommentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *COMMENT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  CommentContext* comment();

  class  RangeContext : public antlr4::ParserRuleContext {
  public:
    RangeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> INT();
    antlr4::tree::TerminalNode* INT(size_t i);
    std::vector<IdContext *> id();
    IdContext* id(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  RangeContext* range();

  class  GroupnamelistContext : public antlr4::ParserRuleContext {
  public:
    GroupnamelistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    GroupnameContext *groupname();
    GroupnamelistContext *groupnamelist();

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
