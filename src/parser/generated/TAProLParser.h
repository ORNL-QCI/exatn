
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
    RuleTaprolsrc = 0, RuleEntry = 1, RuleEntryname = 2, RuleScope = 3, 
    RuleScopename = 4, RuleGroupnamelist = 5, RuleGroupname = 6, RuleCode = 7, 
    RuleLine = 8, RuleStatement = 9, RuleCompositeop = 10, RuleSimpleop = 11, 
    RuleIndex = 12, RuleSubspace = 13, RuleSubspacelist = 14, RuleSubspacename = 15, 
    RuleSpace = 16, RuleSpacelist = 17, RuleSpacename = 18, RuleAssignment = 19, 
    RuleLoad = 20, RuleSave = 21, RuleDestroy = 22, RuleCopy = 23, RuleScale = 24, 
    RuleUnaryop = 25, RuleBinaryop = 26, RuleCompositeproduct = 27, RuleTensornetwork = 28, 
    RuleConjtensor = 29, RuleTensor = 30, RuleTensorname = 31, RuleTensormodelist = 32, 
    RuleTensormode = 33, RuleIndexlist = 34, RuleIndexlabel = 35, RuleRange = 36, 
    RuleId = 37, RuleComplex = 38, RuleReal = 39, RuleString = 40, RuleComment = 41
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
  class EntrynameContext;
  class ScopeContext;
  class ScopenameContext;
  class GroupnamelistContext;
  class GroupnameContext;
  class CodeContext;
  class LineContext;
  class StatementContext;
  class CompositeopContext;
  class SimpleopContext;
  class IndexContext;
  class SubspaceContext;
  class SubspacelistContext;
  class SubspacenameContext;
  class SpaceContext;
  class SpacelistContext;
  class SpacenameContext;
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
  class ConjtensorContext;
  class TensorContext;
  class TensornameContext;
  class TensormodelistContext;
  class TensormodeContext;
  class IndexlistContext;
  class IndexlabelContext;
  class RangeContext;
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

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  TaprolsrcContext* taprolsrc();

  class  EntryContext : public antlr4::ParserRuleContext {
  public:
    EntryContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    EntrynameContext *entryname();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  EntryContext* entry();

  class  EntrynameContext : public antlr4::ParserRuleContext {
  public:
    EntrynameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  EntrynameContext* entryname();

  class  ScopeContext : public antlr4::ParserRuleContext {
  public:
    ScopeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ScopenameContext *scopename();
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
    antlr4::tree::TerminalNode *EOL();
    SubspaceContext *subspace();
    IndexContext *index();
    SimpleopContext *simpleop();
    CompositeopContext *compositeop();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  StatementContext* statement();

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

  class  IndexContext : public antlr4::ParserRuleContext {
  public:
    IndexContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SubspacenameContext *subspacename();
    IndexlistContext *indexlist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  IndexContext* index();

  class  SubspaceContext : public antlr4::ParserRuleContext {
  public:
    SubspaceContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SubspacelistContext *subspacelist();
    SpacenameContext *spacename();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SubspaceContext* subspace();

  class  SubspacelistContext : public antlr4::ParserRuleContext {
  public:
    SubspacelistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SubspacenameContext *subspacename();
    RangeContext *range();
    std::vector<SubspacelistContext *> subspacelist();
    SubspacelistContext* subspacelist(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SubspacelistContext* subspacelist();

  class  SubspacenameContext : public antlr4::ParserRuleContext {
  public:
    SubspacenameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SubspacenameContext* subspacename();

  class  SpaceContext : public antlr4::ParserRuleContext {
  public:
    SpaceContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SpacelistContext *spacelist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SpaceContext* space();

  class  SpacelistContext : public antlr4::ParserRuleContext {
  public:
    SpacelistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SpacenameContext *spacename();
    RangeContext *range();
    std::vector<SpacelistContext *> spacelist();
    SpacelistContext* spacelist(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SpacelistContext* spacelist();

  class  SpacenameContext : public antlr4::ParserRuleContext {
  public:
    SpacenameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  SpacenameContext* spacename();

  class  AssignmentContext : public antlr4::ParserRuleContext {
  public:
    AssignmentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TensorContext *tensor();
    RealContext *real();
    ComplexContext *complex();
    StringContext *string();

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

  class  TensornameContext : public antlr4::ParserRuleContext {
  public:
    TensornameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  TensornameContext* tensorname();

  class  TensormodelistContext : public antlr4::ParserRuleContext {
  public:
    TensormodelistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TensormodeContext *> tensormode();
    TensormodeContext* tensormode(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  TensormodelistContext* tensormodelist();

  class  TensormodeContext : public antlr4::ParserRuleContext {
  public:
    TensormodeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IndexlabelContext *indexlabel();
    RangeContext *range();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  TensormodeContext* tensormode();

  class  IndexlistContext : public antlr4::ParserRuleContext {
  public:
    IndexlistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<IndexlabelContext *> indexlabel();
    IndexlabelContext* indexlabel(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  IndexlistContext* indexlist();

  class  IndexlabelContext : public antlr4::ParserRuleContext {
  public:
    IndexlabelContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  IndexlabelContext* indexlabel();

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
