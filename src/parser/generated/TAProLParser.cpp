
// Generated from TAProL.g4 by ANTLR 4.7.2


#include "TAProLListener.h"

#include "TAProLParser.h"


using namespace antlrcpp;
using namespace taprol;
using namespace antlr4;

TAProLParser::TAProLParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

TAProLParser::~TAProLParser() {
  delete _interpreter;
}

std::string TAProLParser::getGrammarFileName() const {
  return "TAProL.g4";
}

const std::vector<std::string>& TAProLParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& TAProLParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- TaprolsrcContext ------------------------------------------------------------------

TAProLParser::TaprolsrcContext::TaprolsrcContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::EntryContext* TAProLParser::TaprolsrcContext::entry() {
  return getRuleContext<TAProLParser::EntryContext>(0);
}

std::vector<TAProLParser::ScopeContext *> TAProLParser::TaprolsrcContext::scope() {
  return getRuleContexts<TAProLParser::ScopeContext>();
}

TAProLParser::ScopeContext* TAProLParser::TaprolsrcContext::scope(size_t i) {
  return getRuleContext<TAProLParser::ScopeContext>(i);
}

TAProLParser::CodeContext* TAProLParser::TaprolsrcContext::code() {
  return getRuleContext<TAProLParser::CodeContext>(0);
}


size_t TAProLParser::TaprolsrcContext::getRuleIndex() const {
  return TAProLParser::RuleTaprolsrc;
}

void TAProLParser::TaprolsrcContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTaprolsrc(this);
}

void TAProLParser::TaprolsrcContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTaprolsrc(this);
}

TAProLParser::TaprolsrcContext* TAProLParser::taprolsrc() {
  TaprolsrcContext *_localctx = _tracker.createInstance<TaprolsrcContext>(_ctx, getState());
  enterRule(_localctx, 0, TAProLParser::RuleTaprolsrc);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(105);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TAProLParser::T__0: {
        enterOuterAlt(_localctx, 1);
        setState(98);
        entry();
        setState(100); 
        _errHandler->sync(this);
        _la = _input->LA(1);
        do {
          setState(99);
          scope();
          setState(102); 
          _errHandler->sync(this);
          _la = _input->LA(1);
        } while (_la == TAProLParser::T__2);
        break;
      }

      case TAProLParser::EOF:
      case TAProLParser::T__8:
      case TAProLParser::T__11:
      case TAProLParser::T__15:
      case TAProLParser::T__19:
      case TAProLParser::T__21:
      case TAProLParser::T__22:
      case TAProLParser::T__23:
      case TAProLParser::COMMENT:
      case TAProLParser::ID: {
        enterOuterAlt(_localctx, 2);
        setState(104);
        code();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EntryContext ------------------------------------------------------------------

TAProLParser::EntryContext::EntryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::ScopenameContext* TAProLParser::EntryContext::scopename() {
  return getRuleContext<TAProLParser::ScopenameContext>(0);
}


size_t TAProLParser::EntryContext::getRuleIndex() const {
  return TAProLParser::RuleEntry;
}

void TAProLParser::EntryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEntry(this);
}

void TAProLParser::EntryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEntry(this);
}

TAProLParser::EntryContext* TAProLParser::entry() {
  EntryContext *_localctx = _tracker.createInstance<EntryContext>(_ctx, getState());
  enterRule(_localctx, 2, TAProLParser::RuleEntry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(107);
    match(TAProLParser::T__0);
    setState(108);
    match(TAProLParser::T__1);
    setState(109);
    scopename();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ScopeContext ------------------------------------------------------------------

TAProLParser::ScopeContext::ScopeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAProLParser::ScopenameContext *> TAProLParser::ScopeContext::scopename() {
  return getRuleContexts<TAProLParser::ScopenameContext>();
}

TAProLParser::ScopenameContext* TAProLParser::ScopeContext::scopename(size_t i) {
  return getRuleContext<TAProLParser::ScopenameContext>(i);
}

TAProLParser::CodeContext* TAProLParser::ScopeContext::code() {
  return getRuleContext<TAProLParser::CodeContext>(0);
}

TAProLParser::GroupnamelistContext* TAProLParser::ScopeContext::groupnamelist() {
  return getRuleContext<TAProLParser::GroupnamelistContext>(0);
}


size_t TAProLParser::ScopeContext::getRuleIndex() const {
  return TAProLParser::RuleScope;
}

void TAProLParser::ScopeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterScope(this);
}

void TAProLParser::ScopeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitScope(this);
}

TAProLParser::ScopeContext* TAProLParser::scope() {
  ScopeContext *_localctx = _tracker.createInstance<ScopeContext>(_ctx, getState());
  enterRule(_localctx, 4, TAProLParser::RuleScope);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(111);
    match(TAProLParser::T__2);
    setState(112);
    scopename();
    setState(113);
    match(TAProLParser::T__3);
    setState(114);
    match(TAProLParser::T__4);
    setState(116);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAProLParser::ID) {
      setState(115);
      groupnamelist();
    }
    setState(118);
    match(TAProLParser::T__5);
    setState(119);
    code();
    setState(120);
    match(TAProLParser::T__6);
    setState(121);
    scopename();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ScopenameContext ------------------------------------------------------------------

TAProLParser::ScopenameContext::ScopenameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::IdContext* TAProLParser::ScopenameContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
}


size_t TAProLParser::ScopenameContext::getRuleIndex() const {
  return TAProLParser::RuleScopename;
}

void TAProLParser::ScopenameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterScopename(this);
}

void TAProLParser::ScopenameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitScopename(this);
}

TAProLParser::ScopenameContext* TAProLParser::scopename() {
  ScopenameContext *_localctx = _tracker.createInstance<ScopenameContext>(_ctx, getState());
  enterRule(_localctx, 6, TAProLParser::RuleScopename);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(123);
    id();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GroupnamelistContext ------------------------------------------------------------------

TAProLParser::GroupnamelistContext::GroupnamelistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAProLParser::GroupnameContext *> TAProLParser::GroupnamelistContext::groupname() {
  return getRuleContexts<TAProLParser::GroupnameContext>();
}

TAProLParser::GroupnameContext* TAProLParser::GroupnamelistContext::groupname(size_t i) {
  return getRuleContext<TAProLParser::GroupnameContext>(i);
}


size_t TAProLParser::GroupnamelistContext::getRuleIndex() const {
  return TAProLParser::RuleGroupnamelist;
}

void TAProLParser::GroupnamelistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGroupnamelist(this);
}

void TAProLParser::GroupnamelistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGroupnamelist(this);
}

TAProLParser::GroupnamelistContext* TAProLParser::groupnamelist() {
  GroupnamelistContext *_localctx = _tracker.createInstance<GroupnamelistContext>(_ctx, getState());
  enterRule(_localctx, 8, TAProLParser::RuleGroupnamelist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(125);
    groupname();
    setState(130);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TAProLParser::T__7) {
      setState(126);
      match(TAProLParser::T__7);
      setState(127);
      groupname();
      setState(132);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GroupnameContext ------------------------------------------------------------------

TAProLParser::GroupnameContext::GroupnameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::IdContext* TAProLParser::GroupnameContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
}


size_t TAProLParser::GroupnameContext::getRuleIndex() const {
  return TAProLParser::RuleGroupname;
}

void TAProLParser::GroupnameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGroupname(this);
}

void TAProLParser::GroupnameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGroupname(this);
}

TAProLParser::GroupnameContext* TAProLParser::groupname() {
  GroupnameContext *_localctx = _tracker.createInstance<GroupnameContext>(_ctx, getState());
  enterRule(_localctx, 10, TAProLParser::RuleGroupname);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(133);
    id();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CodeContext ------------------------------------------------------------------

TAProLParser::CodeContext::CodeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAProLParser::LineContext *> TAProLParser::CodeContext::line() {
  return getRuleContexts<TAProLParser::LineContext>();
}

TAProLParser::LineContext* TAProLParser::CodeContext::line(size_t i) {
  return getRuleContext<TAProLParser::LineContext>(i);
}


size_t TAProLParser::CodeContext::getRuleIndex() const {
  return TAProLParser::RuleCode;
}

void TAProLParser::CodeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCode(this);
}

void TAProLParser::CodeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCode(this);
}

TAProLParser::CodeContext* TAProLParser::code() {
  CodeContext *_localctx = _tracker.createInstance<CodeContext>(_ctx, getState());
  enterRule(_localctx, 12, TAProLParser::RuleCode);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(138);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << TAProLParser::T__8)
      | (1ULL << TAProLParser::T__11)
      | (1ULL << TAProLParser::T__15)
      | (1ULL << TAProLParser::T__19)
      | (1ULL << TAProLParser::T__21)
      | (1ULL << TAProLParser::T__22)
      | (1ULL << TAProLParser::T__23)
      | (1ULL << TAProLParser::COMMENT)
      | (1ULL << TAProLParser::ID))) != 0)) {
      setState(135);
      line();
      setState(140);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LineContext ------------------------------------------------------------------

TAProLParser::LineContext::LineContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::StatementContext* TAProLParser::LineContext::statement() {
  return getRuleContext<TAProLParser::StatementContext>(0);
}

TAProLParser::CommentContext* TAProLParser::LineContext::comment() {
  return getRuleContext<TAProLParser::CommentContext>(0);
}


size_t TAProLParser::LineContext::getRuleIndex() const {
  return TAProLParser::RuleLine;
}

void TAProLParser::LineContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLine(this);
}

void TAProLParser::LineContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLine(this);
}

TAProLParser::LineContext* TAProLParser::line() {
  LineContext *_localctx = _tracker.createInstance<LineContext>(_ctx, getState());
  enterRule(_localctx, 14, TAProLParser::RuleLine);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(143);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TAProLParser::T__8:
      case TAProLParser::T__11:
      case TAProLParser::T__15:
      case TAProLParser::T__19:
      case TAProLParser::T__21:
      case TAProLParser::T__22:
      case TAProLParser::T__23:
      case TAProLParser::ID: {
        enterOuterAlt(_localctx, 1);
        setState(141);
        statement();
        break;
      }

      case TAProLParser::COMMENT: {
        enterOuterAlt(_localctx, 2);
        setState(142);
        comment();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

TAProLParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::SpaceContext* TAProLParser::StatementContext::space() {
  return getRuleContext<TAProLParser::SpaceContext>(0);
}

TAProLParser::SubspaceContext* TAProLParser::StatementContext::subspace() {
  return getRuleContext<TAProLParser::SubspaceContext>(0);
}

TAProLParser::IndexContext* TAProLParser::StatementContext::index() {
  return getRuleContext<TAProLParser::IndexContext>(0);
}

TAProLParser::SimpleopContext* TAProLParser::StatementContext::simpleop() {
  return getRuleContext<TAProLParser::SimpleopContext>(0);
}

TAProLParser::CompositeopContext* TAProLParser::StatementContext::compositeop() {
  return getRuleContext<TAProLParser::CompositeopContext>(0);
}


size_t TAProLParser::StatementContext::getRuleIndex() const {
  return TAProLParser::RuleStatement;
}

void TAProLParser::StatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatement(this);
}

void TAProLParser::StatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatement(this);
}

TAProLParser::StatementContext* TAProLParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 16, TAProLParser::RuleStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(150);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(145);
      space();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(146);
      subspace();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(147);
      index();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(148);
      simpleop();
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(149);
      compositeop();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SimpleopContext ------------------------------------------------------------------

TAProLParser::SimpleopContext::SimpleopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::AssignContext* TAProLParser::SimpleopContext::assign() {
  return getRuleContext<TAProLParser::AssignContext>(0);
}

TAProLParser::RetrieveContext* TAProLParser::SimpleopContext::retrieve() {
  return getRuleContext<TAProLParser::RetrieveContext>(0);
}

TAProLParser::LoadContext* TAProLParser::SimpleopContext::load() {
  return getRuleContext<TAProLParser::LoadContext>(0);
}

TAProLParser::SaveContext* TAProLParser::SimpleopContext::save() {
  return getRuleContext<TAProLParser::SaveContext>(0);
}

TAProLParser::DestroyContext* TAProLParser::SimpleopContext::destroy() {
  return getRuleContext<TAProLParser::DestroyContext>(0);
}

TAProLParser::NormContext* TAProLParser::SimpleopContext::norm() {
  return getRuleContext<TAProLParser::NormContext>(0);
}

TAProLParser::ScaleContext* TAProLParser::SimpleopContext::scale() {
  return getRuleContext<TAProLParser::ScaleContext>(0);
}

TAProLParser::CopyContext* TAProLParser::SimpleopContext::copy() {
  return getRuleContext<TAProLParser::CopyContext>(0);
}

TAProLParser::AdditionContext* TAProLParser::SimpleopContext::addition() {
  return getRuleContext<TAProLParser::AdditionContext>(0);
}

TAProLParser::ContractionContext* TAProLParser::SimpleopContext::contraction() {
  return getRuleContext<TAProLParser::ContractionContext>(0);
}


size_t TAProLParser::SimpleopContext::getRuleIndex() const {
  return TAProLParser::RuleSimpleop;
}

void TAProLParser::SimpleopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSimpleop(this);
}

void TAProLParser::SimpleopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSimpleop(this);
}

TAProLParser::SimpleopContext* TAProLParser::simpleop() {
  SimpleopContext *_localctx = _tracker.createInstance<SimpleopContext>(_ctx, getState());
  enterRule(_localctx, 18, TAProLParser::RuleSimpleop);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(162);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(152);
      assign();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(153);
      retrieve();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(154);
      load();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(155);
      save();
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(156);
      destroy();
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(157);
      norm();
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(158);
      scale();
      break;
    }

    case 8: {
      enterOuterAlt(_localctx, 8);
      setState(159);
      copy();
      break;
    }

    case 9: {
      enterOuterAlt(_localctx, 9);
      setState(160);
      addition();
      break;
    }

    case 10: {
      enterOuterAlt(_localctx, 10);
      setState(161);
      contraction();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CompositeopContext ------------------------------------------------------------------

TAProLParser::CompositeopContext::CompositeopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::CompositeproductContext* TAProLParser::CompositeopContext::compositeproduct() {
  return getRuleContext<TAProLParser::CompositeproductContext>(0);
}

TAProLParser::TensornetworkContext* TAProLParser::CompositeopContext::tensornetwork() {
  return getRuleContext<TAProLParser::TensornetworkContext>(0);
}


size_t TAProLParser::CompositeopContext::getRuleIndex() const {
  return TAProLParser::RuleCompositeop;
}

void TAProLParser::CompositeopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCompositeop(this);
}

void TAProLParser::CompositeopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCompositeop(this);
}

TAProLParser::CompositeopContext* TAProLParser::compositeop() {
  CompositeopContext *_localctx = _tracker.createInstance<CompositeopContext>(_ctx, getState());
  enterRule(_localctx, 20, TAProLParser::RuleCompositeop);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(166);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(164);
      compositeproduct();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(165);
      tensornetwork();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SpaceContext ------------------------------------------------------------------

TAProLParser::SpaceContext::SpaceContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::NumfieldContext* TAProLParser::SpaceContext::numfield() {
  return getRuleContext<TAProLParser::NumfieldContext>(0);
}

TAProLParser::SpacedeflistContext* TAProLParser::SpaceContext::spacedeflist() {
  return getRuleContext<TAProLParser::SpacedeflistContext>(0);
}


size_t TAProLParser::SpaceContext::getRuleIndex() const {
  return TAProLParser::RuleSpace;
}

void TAProLParser::SpaceContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSpace(this);
}

void TAProLParser::SpaceContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSpace(this);
}

TAProLParser::SpaceContext* TAProLParser::space() {
  SpaceContext *_localctx = _tracker.createInstance<SpaceContext>(_ctx, getState());
  enterRule(_localctx, 22, TAProLParser::RuleSpace);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(168);
    match(TAProLParser::T__8);
    setState(169);
    match(TAProLParser::T__4);
    setState(170);
    numfield();
    setState(171);
    match(TAProLParser::T__5);
    setState(172);
    match(TAProLParser::T__1);
    setState(173);
    spacedeflist();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NumfieldContext ------------------------------------------------------------------

TAProLParser::NumfieldContext::NumfieldContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t TAProLParser::NumfieldContext::getRuleIndex() const {
  return TAProLParser::RuleNumfield;
}

void TAProLParser::NumfieldContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNumfield(this);
}

void TAProLParser::NumfieldContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNumfield(this);
}

TAProLParser::NumfieldContext* TAProLParser::numfield() {
  NumfieldContext *_localctx = _tracker.createInstance<NumfieldContext>(_ctx, getState());
  enterRule(_localctx, 24, TAProLParser::RuleNumfield);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(175);
    _la = _input->LA(1);
    if (!(_la == TAProLParser::T__9

    || _la == TAProLParser::T__10)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SubspaceContext ------------------------------------------------------------------

TAProLParser::SubspaceContext::SubspaceContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::SpacedeflistContext* TAProLParser::SubspaceContext::spacedeflist() {
  return getRuleContext<TAProLParser::SpacedeflistContext>(0);
}

TAProLParser::SpacenameContext* TAProLParser::SubspaceContext::spacename() {
  return getRuleContext<TAProLParser::SpacenameContext>(0);
}


size_t TAProLParser::SubspaceContext::getRuleIndex() const {
  return TAProLParser::RuleSubspace;
}

void TAProLParser::SubspaceContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSubspace(this);
}

void TAProLParser::SubspaceContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSubspace(this);
}

TAProLParser::SubspaceContext* TAProLParser::subspace() {
  SubspaceContext *_localctx = _tracker.createInstance<SubspaceContext>(_ctx, getState());
  enterRule(_localctx, 26, TAProLParser::RuleSubspace);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(177);
    match(TAProLParser::T__11);
    setState(178);
    match(TAProLParser::T__4);
    setState(180);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAProLParser::ID) {
      setState(179);
      spacename();
    }
    setState(182);
    match(TAProLParser::T__5);
    setState(183);
    match(TAProLParser::T__1);
    setState(184);
    spacedeflist();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SpacedeflistContext ------------------------------------------------------------------

TAProLParser::SpacedeflistContext::SpacedeflistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAProLParser::SpacedefContext *> TAProLParser::SpacedeflistContext::spacedef() {
  return getRuleContexts<TAProLParser::SpacedefContext>();
}

TAProLParser::SpacedefContext* TAProLParser::SpacedeflistContext::spacedef(size_t i) {
  return getRuleContext<TAProLParser::SpacedefContext>(i);
}


size_t TAProLParser::SpacedeflistContext::getRuleIndex() const {
  return TAProLParser::RuleSpacedeflist;
}

void TAProLParser::SpacedeflistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSpacedeflist(this);
}

void TAProLParser::SpacedeflistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSpacedeflist(this);
}

TAProLParser::SpacedeflistContext* TAProLParser::spacedeflist() {
  SpacedeflistContext *_localctx = _tracker.createInstance<SpacedeflistContext>(_ctx, getState());
  enterRule(_localctx, 28, TAProLParser::RuleSpacedeflist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(186);
    spacedef();
    setState(191);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TAProLParser::T__7) {
      setState(187);
      match(TAProLParser::T__7);
      setState(188);
      spacedef();
      setState(193);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SpacedefContext ------------------------------------------------------------------

TAProLParser::SpacedefContext::SpacedefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::SpacenameContext* TAProLParser::SpacedefContext::spacename() {
  return getRuleContext<TAProLParser::SpacenameContext>(0);
}

TAProLParser::RangeContext* TAProLParser::SpacedefContext::range() {
  return getRuleContext<TAProLParser::RangeContext>(0);
}


size_t TAProLParser::SpacedefContext::getRuleIndex() const {
  return TAProLParser::RuleSpacedef;
}

void TAProLParser::SpacedefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSpacedef(this);
}

void TAProLParser::SpacedefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSpacedef(this);
}

TAProLParser::SpacedefContext* TAProLParser::spacedef() {
  SpacedefContext *_localctx = _tracker.createInstance<SpacedefContext>(_ctx, getState());
  enterRule(_localctx, 30, TAProLParser::RuleSpacedef);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(194);
    spacename();
    setState(195);
    match(TAProLParser::T__12);
    setState(196);
    range();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SpacenameContext ------------------------------------------------------------------

TAProLParser::SpacenameContext::SpacenameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::IdContext* TAProLParser::SpacenameContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
}


size_t TAProLParser::SpacenameContext::getRuleIndex() const {
  return TAProLParser::RuleSpacename;
}

void TAProLParser::SpacenameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSpacename(this);
}

void TAProLParser::SpacenameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSpacename(this);
}

TAProLParser::SpacenameContext* TAProLParser::spacename() {
  SpacenameContext *_localctx = _tracker.createInstance<SpacenameContext>(_ctx, getState());
  enterRule(_localctx, 32, TAProLParser::RuleSpacename);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(198);
    id();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RangeContext ------------------------------------------------------------------

TAProLParser::RangeContext::RangeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::LowerboundContext* TAProLParser::RangeContext::lowerbound() {
  return getRuleContext<TAProLParser::LowerboundContext>(0);
}

TAProLParser::UpperboundContext* TAProLParser::RangeContext::upperbound() {
  return getRuleContext<TAProLParser::UpperboundContext>(0);
}


size_t TAProLParser::RangeContext::getRuleIndex() const {
  return TAProLParser::RuleRange;
}

void TAProLParser::RangeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRange(this);
}

void TAProLParser::RangeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRange(this);
}

TAProLParser::RangeContext* TAProLParser::range() {
  RangeContext *_localctx = _tracker.createInstance<RangeContext>(_ctx, getState());
  enterRule(_localctx, 34, TAProLParser::RuleRange);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(200);
    match(TAProLParser::T__13);
    setState(201);
    lowerbound();
    setState(202);
    match(TAProLParser::T__1);
    setState(203);
    upperbound();
    setState(204);
    match(TAProLParser::T__14);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LowerboundContext ------------------------------------------------------------------

TAProLParser::LowerboundContext::LowerboundContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAProLParser::LowerboundContext::INT() {
  return getToken(TAProLParser::INT, 0);
}

TAProLParser::IdContext* TAProLParser::LowerboundContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
}


size_t TAProLParser::LowerboundContext::getRuleIndex() const {
  return TAProLParser::RuleLowerbound;
}

void TAProLParser::LowerboundContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLowerbound(this);
}

void TAProLParser::LowerboundContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLowerbound(this);
}

TAProLParser::LowerboundContext* TAProLParser::lowerbound() {
  LowerboundContext *_localctx = _tracker.createInstance<LowerboundContext>(_ctx, getState());
  enterRule(_localctx, 36, TAProLParser::RuleLowerbound);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(208);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TAProLParser::INT: {
        enterOuterAlt(_localctx, 1);
        setState(206);
        match(TAProLParser::INT);
        break;
      }

      case TAProLParser::ID: {
        enterOuterAlt(_localctx, 2);
        setState(207);
        id();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UpperboundContext ------------------------------------------------------------------

TAProLParser::UpperboundContext::UpperboundContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAProLParser::UpperboundContext::INT() {
  return getToken(TAProLParser::INT, 0);
}

TAProLParser::IdContext* TAProLParser::UpperboundContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
}


size_t TAProLParser::UpperboundContext::getRuleIndex() const {
  return TAProLParser::RuleUpperbound;
}

void TAProLParser::UpperboundContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUpperbound(this);
}

void TAProLParser::UpperboundContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUpperbound(this);
}

TAProLParser::UpperboundContext* TAProLParser::upperbound() {
  UpperboundContext *_localctx = _tracker.createInstance<UpperboundContext>(_ctx, getState());
  enterRule(_localctx, 38, TAProLParser::RuleUpperbound);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(212);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TAProLParser::INT: {
        enterOuterAlt(_localctx, 1);
        setState(210);
        match(TAProLParser::INT);
        break;
      }

      case TAProLParser::ID: {
        enterOuterAlt(_localctx, 2);
        setState(211);
        id();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IndexContext ------------------------------------------------------------------

TAProLParser::IndexContext::IndexContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::SpacenameContext* TAProLParser::IndexContext::spacename() {
  return getRuleContext<TAProLParser::SpacenameContext>(0);
}

TAProLParser::IndexlistContext* TAProLParser::IndexContext::indexlist() {
  return getRuleContext<TAProLParser::IndexlistContext>(0);
}


size_t TAProLParser::IndexContext::getRuleIndex() const {
  return TAProLParser::RuleIndex;
}

void TAProLParser::IndexContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIndex(this);
}

void TAProLParser::IndexContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIndex(this);
}

TAProLParser::IndexContext* TAProLParser::index() {
  IndexContext *_localctx = _tracker.createInstance<IndexContext>(_ctx, getState());
  enterRule(_localctx, 40, TAProLParser::RuleIndex);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(214);
    match(TAProLParser::T__15);
    setState(215);
    match(TAProLParser::T__4);
    setState(216);
    spacename();
    setState(217);
    match(TAProLParser::T__5);
    setState(218);
    match(TAProLParser::T__1);
    setState(219);
    indexlist();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IndexlistContext ------------------------------------------------------------------

TAProLParser::IndexlistContext::IndexlistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAProLParser::IndexnameContext *> TAProLParser::IndexlistContext::indexname() {
  return getRuleContexts<TAProLParser::IndexnameContext>();
}

TAProLParser::IndexnameContext* TAProLParser::IndexlistContext::indexname(size_t i) {
  return getRuleContext<TAProLParser::IndexnameContext>(i);
}


size_t TAProLParser::IndexlistContext::getRuleIndex() const {
  return TAProLParser::RuleIndexlist;
}

void TAProLParser::IndexlistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIndexlist(this);
}

void TAProLParser::IndexlistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIndexlist(this);
}

TAProLParser::IndexlistContext* TAProLParser::indexlist() {
  IndexlistContext *_localctx = _tracker.createInstance<IndexlistContext>(_ctx, getState());
  enterRule(_localctx, 42, TAProLParser::RuleIndexlist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(221);
    indexname();
    setState(226);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TAProLParser::T__7) {
      setState(222);
      match(TAProLParser::T__7);
      setState(223);
      indexname();
      setState(228);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IndexnameContext ------------------------------------------------------------------

TAProLParser::IndexnameContext::IndexnameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::IdContext* TAProLParser::IndexnameContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
}


size_t TAProLParser::IndexnameContext::getRuleIndex() const {
  return TAProLParser::RuleIndexname;
}

void TAProLParser::IndexnameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIndexname(this);
}

void TAProLParser::IndexnameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIndexname(this);
}

TAProLParser::IndexnameContext* TAProLParser::indexname() {
  IndexnameContext *_localctx = _tracker.createInstance<IndexnameContext>(_ctx, getState());
  enterRule(_localctx, 44, TAProLParser::RuleIndexname);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(229);
    id();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AssignContext ------------------------------------------------------------------

TAProLParser::AssignContext::AssignContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::TensorContext* TAProLParser::AssignContext::tensor() {
  return getRuleContext<TAProLParser::TensorContext>(0);
}

TAProLParser::RealContext* TAProLParser::AssignContext::real() {
  return getRuleContext<TAProLParser::RealContext>(0);
}

TAProLParser::ComplexContext* TAProLParser::AssignContext::complex() {
  return getRuleContext<TAProLParser::ComplexContext>(0);
}

TAProLParser::DatacontainerContext* TAProLParser::AssignContext::datacontainer() {
  return getRuleContext<TAProLParser::DatacontainerContext>(0);
}

TAProLParser::MethodnameContext* TAProLParser::AssignContext::methodname() {
  return getRuleContext<TAProLParser::MethodnameContext>(0);
}


size_t TAProLParser::AssignContext::getRuleIndex() const {
  return TAProLParser::RuleAssign;
}

void TAProLParser::AssignContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssign(this);
}

void TAProLParser::AssignContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssign(this);
}

TAProLParser::AssignContext* TAProLParser::assign() {
  AssignContext *_localctx = _tracker.createInstance<AssignContext>(_ctx, getState());
  enterRule(_localctx, 46, TAProLParser::RuleAssign);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(259);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 15, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(231);
      tensor();
      setState(232);
      match(TAProLParser::T__12);
      setState(233);
      match(TAProLParser::T__16);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(235);
      tensor();
      setState(236);
      match(TAProLParser::T__12);
      setState(239);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case TAProLParser::REAL:
        case TAProLParser::FLOAT: {
          setState(237);
          real();
          break;
        }

        case TAProLParser::T__31: {
          setState(238);
          complex();
          break;
        }

      default:
        throw NoViableAltException(this);
      }
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(241);
      tensor();
      setState(242);
      match(TAProLParser::T__12);
      setState(243);
      datacontainer();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(245);
      tensor();
      setState(246);
      match(TAProLParser::T__12);
      setState(247);
      match(TAProLParser::T__17);
      setState(248);
      match(TAProLParser::T__4);
      setState(249);
      methodname();
      setState(250);
      match(TAProLParser::T__5);
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(252);
      tensor();
      setState(253);
      match(TAProLParser::T__18);
      setState(254);
      match(TAProLParser::T__17);
      setState(255);
      match(TAProLParser::T__4);
      setState(256);
      methodname();
      setState(257);
      match(TAProLParser::T__5);
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DatacontainerContext ------------------------------------------------------------------

TAProLParser::DatacontainerContext::DatacontainerContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::IdContext* TAProLParser::DatacontainerContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
}


size_t TAProLParser::DatacontainerContext::getRuleIndex() const {
  return TAProLParser::RuleDatacontainer;
}

void TAProLParser::DatacontainerContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDatacontainer(this);
}

void TAProLParser::DatacontainerContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDatacontainer(this);
}

TAProLParser::DatacontainerContext* TAProLParser::datacontainer() {
  DatacontainerContext *_localctx = _tracker.createInstance<DatacontainerContext>(_ctx, getState());
  enterRule(_localctx, 48, TAProLParser::RuleDatacontainer);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(261);
    id();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MethodnameContext ------------------------------------------------------------------

TAProLParser::MethodnameContext::MethodnameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::StringContext* TAProLParser::MethodnameContext::string() {
  return getRuleContext<TAProLParser::StringContext>(0);
}


size_t TAProLParser::MethodnameContext::getRuleIndex() const {
  return TAProLParser::RuleMethodname;
}

void TAProLParser::MethodnameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMethodname(this);
}

void TAProLParser::MethodnameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMethodname(this);
}

TAProLParser::MethodnameContext* TAProLParser::methodname() {
  MethodnameContext *_localctx = _tracker.createInstance<MethodnameContext>(_ctx, getState());
  enterRule(_localctx, 50, TAProLParser::RuleMethodname);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(263);
    string();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RetrieveContext ------------------------------------------------------------------

TAProLParser::RetrieveContext::RetrieveContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::DatacontainerContext* TAProLParser::RetrieveContext::datacontainer() {
  return getRuleContext<TAProLParser::DatacontainerContext>(0);
}

TAProLParser::TensornameContext* TAProLParser::RetrieveContext::tensorname() {
  return getRuleContext<TAProLParser::TensornameContext>(0);
}

TAProLParser::TensorContext* TAProLParser::RetrieveContext::tensor() {
  return getRuleContext<TAProLParser::TensorContext>(0);
}


size_t TAProLParser::RetrieveContext::getRuleIndex() const {
  return TAProLParser::RuleRetrieve;
}

void TAProLParser::RetrieveContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRetrieve(this);
}

void TAProLParser::RetrieveContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRetrieve(this);
}

TAProLParser::RetrieveContext* TAProLParser::retrieve() {
  RetrieveContext *_localctx = _tracker.createInstance<RetrieveContext>(_ctx, getState());
  enterRule(_localctx, 52, TAProLParser::RuleRetrieve);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(273);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 16, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(265);
      datacontainer();
      setState(266);
      match(TAProLParser::T__12);
      setState(267);
      tensorname();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(269);
      datacontainer();
      setState(270);
      match(TAProLParser::T__12);
      setState(271);
      tensor();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LoadContext ------------------------------------------------------------------

TAProLParser::LoadContext::LoadContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::TagnameContext* TAProLParser::LoadContext::tagname() {
  return getRuleContext<TAProLParser::TagnameContext>(0);
}

TAProLParser::TensorContext* TAProLParser::LoadContext::tensor() {
  return getRuleContext<TAProLParser::TensorContext>(0);
}

TAProLParser::TensornameContext* TAProLParser::LoadContext::tensorname() {
  return getRuleContext<TAProLParser::TensornameContext>(0);
}


size_t TAProLParser::LoadContext::getRuleIndex() const {
  return TAProLParser::RuleLoad;
}

void TAProLParser::LoadContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLoad(this);
}

void TAProLParser::LoadContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLoad(this);
}

TAProLParser::LoadContext* TAProLParser::load() {
  LoadContext *_localctx = _tracker.createInstance<LoadContext>(_ctx, getState());
  enterRule(_localctx, 54, TAProLParser::RuleLoad);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(275);
    match(TAProLParser::T__19);
    setState(278);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 17, _ctx)) {
    case 1: {
      setState(276);
      tensor();
      break;
    }

    case 2: {
      setState(277);
      tensorname();
      break;
    }

    }
    setState(280);
    match(TAProLParser::T__1);
    setState(281);
    match(TAProLParser::T__20);
    setState(282);
    match(TAProLParser::T__4);
    setState(283);
    tagname();
    setState(284);
    match(TAProLParser::T__5);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SaveContext ------------------------------------------------------------------

TAProLParser::SaveContext::SaveContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::TagnameContext* TAProLParser::SaveContext::tagname() {
  return getRuleContext<TAProLParser::TagnameContext>(0);
}

TAProLParser::TensorContext* TAProLParser::SaveContext::tensor() {
  return getRuleContext<TAProLParser::TensorContext>(0);
}

TAProLParser::TensornameContext* TAProLParser::SaveContext::tensorname() {
  return getRuleContext<TAProLParser::TensornameContext>(0);
}


size_t TAProLParser::SaveContext::getRuleIndex() const {
  return TAProLParser::RuleSave;
}

void TAProLParser::SaveContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSave(this);
}

void TAProLParser::SaveContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSave(this);
}

TAProLParser::SaveContext* TAProLParser::save() {
  SaveContext *_localctx = _tracker.createInstance<SaveContext>(_ctx, getState());
  enterRule(_localctx, 56, TAProLParser::RuleSave);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(286);
    match(TAProLParser::T__21);
    setState(289);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 18, _ctx)) {
    case 1: {
      setState(287);
      tensor();
      break;
    }

    case 2: {
      setState(288);
      tensorname();
      break;
    }

    }
    setState(291);
    match(TAProLParser::T__1);
    setState(292);
    match(TAProLParser::T__20);
    setState(293);
    match(TAProLParser::T__4);
    setState(294);
    tagname();
    setState(295);
    match(TAProLParser::T__5);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TagnameContext ------------------------------------------------------------------

TAProLParser::TagnameContext::TagnameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::StringContext* TAProLParser::TagnameContext::string() {
  return getRuleContext<TAProLParser::StringContext>(0);
}


size_t TAProLParser::TagnameContext::getRuleIndex() const {
  return TAProLParser::RuleTagname;
}

void TAProLParser::TagnameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTagname(this);
}

void TAProLParser::TagnameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTagname(this);
}

TAProLParser::TagnameContext* TAProLParser::tagname() {
  TagnameContext *_localctx = _tracker.createInstance<TagnameContext>(_ctx, getState());
  enterRule(_localctx, 58, TAProLParser::RuleTagname);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(297);
    string();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DestroyContext ------------------------------------------------------------------

TAProLParser::DestroyContext::DestroyContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::TensornameContext* TAProLParser::DestroyContext::tensorname() {
  return getRuleContext<TAProLParser::TensornameContext>(0);
}

TAProLParser::TensorContext* TAProLParser::DestroyContext::tensor() {
  return getRuleContext<TAProLParser::TensorContext>(0);
}

TAProLParser::TensorlistContext* TAProLParser::DestroyContext::tensorlist() {
  return getRuleContext<TAProLParser::TensorlistContext>(0);
}


size_t TAProLParser::DestroyContext::getRuleIndex() const {
  return TAProLParser::RuleDestroy;
}

void TAProLParser::DestroyContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDestroy(this);
}

void TAProLParser::DestroyContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDestroy(this);
}

TAProLParser::DestroyContext* TAProLParser::destroy() {
  DestroyContext *_localctx = _tracker.createInstance<DestroyContext>(_ctx, getState());
  enterRule(_localctx, 60, TAProLParser::RuleDestroy);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(305);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 19, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(299);
      match(TAProLParser::T__22);
      setState(300);
      tensorname();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(301);
      match(TAProLParser::T__22);
      setState(302);
      tensor();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(303);
      match(TAProLParser::T__23);
      setState(304);
      tensorlist();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TensorlistContext ------------------------------------------------------------------

TAProLParser::TensorlistContext::TensorlistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAProLParser::TensornameContext *> TAProLParser::TensorlistContext::tensorname() {
  return getRuleContexts<TAProLParser::TensornameContext>();
}

TAProLParser::TensornameContext* TAProLParser::TensorlistContext::tensorname(size_t i) {
  return getRuleContext<TAProLParser::TensornameContext>(i);
}

std::vector<TAProLParser::TensorContext *> TAProLParser::TensorlistContext::tensor() {
  return getRuleContexts<TAProLParser::TensorContext>();
}

TAProLParser::TensorContext* TAProLParser::TensorlistContext::tensor(size_t i) {
  return getRuleContext<TAProLParser::TensorContext>(i);
}


size_t TAProLParser::TensorlistContext::getRuleIndex() const {
  return TAProLParser::RuleTensorlist;
}

void TAProLParser::TensorlistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTensorlist(this);
}

void TAProLParser::TensorlistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTensorlist(this);
}

TAProLParser::TensorlistContext* TAProLParser::tensorlist() {
  TensorlistContext *_localctx = _tracker.createInstance<TensorlistContext>(_ctx, getState());
  enterRule(_localctx, 62, TAProLParser::RuleTensorlist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(309);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 20, _ctx)) {
    case 1: {
      setState(307);
      tensorname();
      break;
    }

    case 2: {
      setState(308);
      tensor();
      break;
    }

    }
    setState(318);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TAProLParser::T__7) {
      setState(311);
      match(TAProLParser::T__7);
      setState(314);
      _errHandler->sync(this);
      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 21, _ctx)) {
      case 1: {
        setState(312);
        tensorname();
        break;
      }

      case 2: {
        setState(313);
        tensor();
        break;
      }

      }
      setState(320);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NormContext ------------------------------------------------------------------

TAProLParser::NormContext::NormContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::ScalarContext* TAProLParser::NormContext::scalar() {
  return getRuleContext<TAProLParser::ScalarContext>(0);
}

TAProLParser::TensornameContext* TAProLParser::NormContext::tensorname() {
  return getRuleContext<TAProLParser::TensornameContext>(0);
}

TAProLParser::TensorContext* TAProLParser::NormContext::tensor() {
  return getRuleContext<TAProLParser::TensorContext>(0);
}


size_t TAProLParser::NormContext::getRuleIndex() const {
  return TAProLParser::RuleNorm;
}

void TAProLParser::NormContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNorm(this);
}

void TAProLParser::NormContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNorm(this);
}

TAProLParser::NormContext* TAProLParser::norm() {
  NormContext *_localctx = _tracker.createInstance<NormContext>(_ctx, getState());
  enterRule(_localctx, 64, TAProLParser::RuleNorm);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(351);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 26, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(321);
      scalar();
      setState(322);
      match(TAProLParser::T__12);
      setState(323);
      match(TAProLParser::T__24);
      setState(324);
      match(TAProLParser::T__4);
      setState(327);
      _errHandler->sync(this);
      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 23, _ctx)) {
      case 1: {
        setState(325);
        tensorname();
        break;
      }

      case 2: {
        setState(326);
        tensor();
        break;
      }

      }
      setState(329);
      match(TAProLParser::T__5);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(331);
      scalar();
      setState(332);
      match(TAProLParser::T__12);
      setState(333);
      match(TAProLParser::T__25);
      setState(334);
      match(TAProLParser::T__4);
      setState(337);
      _errHandler->sync(this);
      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx)) {
      case 1: {
        setState(335);
        tensorname();
        break;
      }

      case 2: {
        setState(336);
        tensor();
        break;
      }

      }
      setState(339);
      match(TAProLParser::T__5);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(341);
      scalar();
      setState(342);
      match(TAProLParser::T__12);
      setState(343);
      match(TAProLParser::T__26);
      setState(344);
      match(TAProLParser::T__4);
      setState(347);
      _errHandler->sync(this);
      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 25, _ctx)) {
      case 1: {
        setState(345);
        tensorname();
        break;
      }

      case 2: {
        setState(346);
        tensor();
        break;
      }

      }
      setState(349);
      match(TAProLParser::T__5);
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ScalarContext ------------------------------------------------------------------

TAProLParser::ScalarContext::ScalarContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::IdContext* TAProLParser::ScalarContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
}


size_t TAProLParser::ScalarContext::getRuleIndex() const {
  return TAProLParser::RuleScalar;
}

void TAProLParser::ScalarContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterScalar(this);
}

void TAProLParser::ScalarContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitScalar(this);
}

TAProLParser::ScalarContext* TAProLParser::scalar() {
  ScalarContext *_localctx = _tracker.createInstance<ScalarContext>(_ctx, getState());
  enterRule(_localctx, 66, TAProLParser::RuleScalar);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(353);
    id();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ScaleContext ------------------------------------------------------------------

TAProLParser::ScaleContext::ScaleContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::TensorContext* TAProLParser::ScaleContext::tensor() {
  return getRuleContext<TAProLParser::TensorContext>(0);
}

TAProLParser::PrefactorContext* TAProLParser::ScaleContext::prefactor() {
  return getRuleContext<TAProLParser::PrefactorContext>(0);
}


size_t TAProLParser::ScaleContext::getRuleIndex() const {
  return TAProLParser::RuleScale;
}

void TAProLParser::ScaleContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterScale(this);
}

void TAProLParser::ScaleContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitScale(this);
}

TAProLParser::ScaleContext* TAProLParser::scale() {
  ScaleContext *_localctx = _tracker.createInstance<ScaleContext>(_ctx, getState());
  enterRule(_localctx, 68, TAProLParser::RuleScale);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(355);
    tensor();
    setState(356);
    match(TAProLParser::T__27);
    setState(357);
    prefactor();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PrefactorContext ------------------------------------------------------------------

TAProLParser::PrefactorContext::PrefactorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::RealContext* TAProLParser::PrefactorContext::real() {
  return getRuleContext<TAProLParser::RealContext>(0);
}

TAProLParser::ComplexContext* TAProLParser::PrefactorContext::complex() {
  return getRuleContext<TAProLParser::ComplexContext>(0);
}

TAProLParser::IdContext* TAProLParser::PrefactorContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
}


size_t TAProLParser::PrefactorContext::getRuleIndex() const {
  return TAProLParser::RulePrefactor;
}

void TAProLParser::PrefactorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPrefactor(this);
}

void TAProLParser::PrefactorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPrefactor(this);
}

TAProLParser::PrefactorContext* TAProLParser::prefactor() {
  PrefactorContext *_localctx = _tracker.createInstance<PrefactorContext>(_ctx, getState());
  enterRule(_localctx, 70, TAProLParser::RulePrefactor);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(362);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TAProLParser::REAL:
      case TAProLParser::FLOAT: {
        enterOuterAlt(_localctx, 1);
        setState(359);
        real();
        break;
      }

      case TAProLParser::T__31: {
        enterOuterAlt(_localctx, 2);
        setState(360);
        complex();
        break;
      }

      case TAProLParser::ID: {
        enterOuterAlt(_localctx, 3);
        setState(361);
        id();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CopyContext ------------------------------------------------------------------

TAProLParser::CopyContext::CopyContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAProLParser::TensorContext *> TAProLParser::CopyContext::tensor() {
  return getRuleContexts<TAProLParser::TensorContext>();
}

TAProLParser::TensorContext* TAProLParser::CopyContext::tensor(size_t i) {
  return getRuleContext<TAProLParser::TensorContext>(i);
}


size_t TAProLParser::CopyContext::getRuleIndex() const {
  return TAProLParser::RuleCopy;
}

void TAProLParser::CopyContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCopy(this);
}

void TAProLParser::CopyContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCopy(this);
}

TAProLParser::CopyContext* TAProLParser::copy() {
  CopyContext *_localctx = _tracker.createInstance<CopyContext>(_ctx, getState());
  enterRule(_localctx, 72, TAProLParser::RuleCopy);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(364);
    tensor();
    setState(365);
    match(TAProLParser::T__12);
    setState(366);
    tensor();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AdditionContext ------------------------------------------------------------------

TAProLParser::AdditionContext::AdditionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAProLParser::TensorContext *> TAProLParser::AdditionContext::tensor() {
  return getRuleContexts<TAProLParser::TensorContext>();
}

TAProLParser::TensorContext* TAProLParser::AdditionContext::tensor(size_t i) {
  return getRuleContext<TAProLParser::TensorContext>(i);
}

TAProLParser::ConjtensorContext* TAProLParser::AdditionContext::conjtensor() {
  return getRuleContext<TAProLParser::ConjtensorContext>(0);
}

TAProLParser::PrefactorContext* TAProLParser::AdditionContext::prefactor() {
  return getRuleContext<TAProLParser::PrefactorContext>(0);
}


size_t TAProLParser::AdditionContext::getRuleIndex() const {
  return TAProLParser::RuleAddition;
}

void TAProLParser::AdditionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAddition(this);
}

void TAProLParser::AdditionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAddition(this);
}

TAProLParser::AdditionContext* TAProLParser::addition() {
  AdditionContext *_localctx = _tracker.createInstance<AdditionContext>(_ctx, getState());
  enterRule(_localctx, 74, TAProLParser::RuleAddition);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(368);
    tensor();
    setState(369);
    match(TAProLParser::T__28);
    setState(372);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 28, _ctx)) {
    case 1: {
      setState(370);
      tensor();
      break;
    }

    case 2: {
      setState(371);
      conjtensor();
      break;
    }

    }
    setState(376);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAProLParser::T__29) {
      setState(374);
      match(TAProLParser::T__29);
      setState(375);
      prefactor();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ContractionContext ------------------------------------------------------------------

TAProLParser::ContractionContext::ContractionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAProLParser::TensorContext *> TAProLParser::ContractionContext::tensor() {
  return getRuleContexts<TAProLParser::TensorContext>();
}

TAProLParser::TensorContext* TAProLParser::ContractionContext::tensor(size_t i) {
  return getRuleContext<TAProLParser::TensorContext>(i);
}

std::vector<TAProLParser::ConjtensorContext *> TAProLParser::ContractionContext::conjtensor() {
  return getRuleContexts<TAProLParser::ConjtensorContext>();
}

TAProLParser::ConjtensorContext* TAProLParser::ContractionContext::conjtensor(size_t i) {
  return getRuleContext<TAProLParser::ConjtensorContext>(i);
}

TAProLParser::PrefactorContext* TAProLParser::ContractionContext::prefactor() {
  return getRuleContext<TAProLParser::PrefactorContext>(0);
}


size_t TAProLParser::ContractionContext::getRuleIndex() const {
  return TAProLParser::RuleContraction;
}

void TAProLParser::ContractionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterContraction(this);
}

void TAProLParser::ContractionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitContraction(this);
}

TAProLParser::ContractionContext* TAProLParser::contraction() {
  ContractionContext *_localctx = _tracker.createInstance<ContractionContext>(_ctx, getState());
  enterRule(_localctx, 76, TAProLParser::RuleContraction);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(378);
    tensor();
    setState(379);
    match(TAProLParser::T__28);
    setState(382);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 30, _ctx)) {
    case 1: {
      setState(380);
      tensor();
      break;
    }

    case 2: {
      setState(381);
      conjtensor();
      break;
    }

    }
    setState(384);
    match(TAProLParser::T__29);
    setState(387);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 31, _ctx)) {
    case 1: {
      setState(385);
      tensor();
      break;
    }

    case 2: {
      setState(386);
      conjtensor();
      break;
    }

    }
    setState(391);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAProLParser::T__29) {
      setState(389);
      match(TAProLParser::T__29);
      setState(390);
      prefactor();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CompositeproductContext ------------------------------------------------------------------

TAProLParser::CompositeproductContext::CompositeproductContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAProLParser::TensorContext *> TAProLParser::CompositeproductContext::tensor() {
  return getRuleContexts<TAProLParser::TensorContext>();
}

TAProLParser::TensorContext* TAProLParser::CompositeproductContext::tensor(size_t i) {
  return getRuleContext<TAProLParser::TensorContext>(i);
}

std::vector<TAProLParser::ConjtensorContext *> TAProLParser::CompositeproductContext::conjtensor() {
  return getRuleContexts<TAProLParser::ConjtensorContext>();
}

TAProLParser::ConjtensorContext* TAProLParser::CompositeproductContext::conjtensor(size_t i) {
  return getRuleContext<TAProLParser::ConjtensorContext>(i);
}

TAProLParser::PrefactorContext* TAProLParser::CompositeproductContext::prefactor() {
  return getRuleContext<TAProLParser::PrefactorContext>(0);
}


size_t TAProLParser::CompositeproductContext::getRuleIndex() const {
  return TAProLParser::RuleCompositeproduct;
}

void TAProLParser::CompositeproductContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCompositeproduct(this);
}

void TAProLParser::CompositeproductContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCompositeproduct(this);
}

TAProLParser::CompositeproductContext* TAProLParser::compositeproduct() {
  CompositeproductContext *_localctx = _tracker.createInstance<CompositeproductContext>(_ctx, getState());
  enterRule(_localctx, 78, TAProLParser::RuleCompositeproduct);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(393);
    tensor();
    setState(394);
    match(TAProLParser::T__28);
    setState(397);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 33, _ctx)) {
    case 1: {
      setState(395);
      tensor();
      break;
    }

    case 2: {
      setState(396);
      conjtensor();
      break;
    }

    }
    setState(399);
    match(TAProLParser::T__29);
    setState(402);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 34, _ctx)) {
    case 1: {
      setState(400);
      tensor();
      break;
    }

    case 2: {
      setState(401);
      conjtensor();
      break;
    }

    }
    setState(409); 
    _errHandler->sync(this);
    alt = 1;
    do {
      switch (alt) {
        case 1: {
              setState(404);
              match(TAProLParser::T__29);
              setState(407);
              _errHandler->sync(this);
              switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 35, _ctx)) {
              case 1: {
                setState(405);
                tensor();
                break;
              }

              case 2: {
                setState(406);
                conjtensor();
                break;
              }

              }
              break;
            }

      default:
        throw NoViableAltException(this);
      }
      setState(411); 
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 36, _ctx);
    } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
    setState(415);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAProLParser::T__29) {
      setState(413);
      match(TAProLParser::T__29);
      setState(414);
      prefactor();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TensornetworkContext ------------------------------------------------------------------

TAProLParser::TensornetworkContext::TensornetworkContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAProLParser::TensorContext *> TAProLParser::TensornetworkContext::tensor() {
  return getRuleContexts<TAProLParser::TensorContext>();
}

TAProLParser::TensorContext* TAProLParser::TensornetworkContext::tensor(size_t i) {
  return getRuleContext<TAProLParser::TensorContext>(i);
}

std::vector<TAProLParser::ConjtensorContext *> TAProLParser::TensornetworkContext::conjtensor() {
  return getRuleContexts<TAProLParser::ConjtensorContext>();
}

TAProLParser::ConjtensorContext* TAProLParser::TensornetworkContext::conjtensor(size_t i) {
  return getRuleContext<TAProLParser::ConjtensorContext>(i);
}


size_t TAProLParser::TensornetworkContext::getRuleIndex() const {
  return TAProLParser::RuleTensornetwork;
}

void TAProLParser::TensornetworkContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTensornetwork(this);
}

void TAProLParser::TensornetworkContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTensornetwork(this);
}

TAProLParser::TensornetworkContext* TAProLParser::tensornetwork() {
  TensornetworkContext *_localctx = _tracker.createInstance<TensornetworkContext>(_ctx, getState());
  enterRule(_localctx, 80, TAProLParser::RuleTensornetwork);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(417);
    tensor();
    setState(418);
    match(TAProLParser::T__18);
    setState(421);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 38, _ctx)) {
    case 1: {
      setState(419);
      tensor();
      break;
    }

    case 2: {
      setState(420);
      conjtensor();
      break;
    }

    }
    setState(428); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(423);
      match(TAProLParser::T__29);
      setState(426);
      _errHandler->sync(this);
      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 39, _ctx)) {
      case 1: {
        setState(424);
        tensor();
        break;
      }

      case 2: {
        setState(425);
        conjtensor();
        break;
      }

      }
      setState(430); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == TAProLParser::T__29);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TensorContext ------------------------------------------------------------------

TAProLParser::TensorContext::TensorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::TensornameContext* TAProLParser::TensorContext::tensorname() {
  return getRuleContext<TAProLParser::TensornameContext>(0);
}

TAProLParser::IndexlistContext* TAProLParser::TensorContext::indexlist() {
  return getRuleContext<TAProLParser::IndexlistContext>(0);
}


size_t TAProLParser::TensorContext::getRuleIndex() const {
  return TAProLParser::RuleTensor;
}

void TAProLParser::TensorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTensor(this);
}

void TAProLParser::TensorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTensor(this);
}

TAProLParser::TensorContext* TAProLParser::tensor() {
  TensorContext *_localctx = _tracker.createInstance<TensorContext>(_ctx, getState());
  enterRule(_localctx, 82, TAProLParser::RuleTensor);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(432);
    tensorname();
    setState(433);
    match(TAProLParser::T__4);
    setState(435);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAProLParser::ID) {
      setState(434);
      indexlist();
    }
    setState(437);
    match(TAProLParser::T__5);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ConjtensorContext ------------------------------------------------------------------

TAProLParser::ConjtensorContext::ConjtensorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::TensornameContext* TAProLParser::ConjtensorContext::tensorname() {
  return getRuleContext<TAProLParser::TensornameContext>(0);
}

TAProLParser::IndexlistContext* TAProLParser::ConjtensorContext::indexlist() {
  return getRuleContext<TAProLParser::IndexlistContext>(0);
}


size_t TAProLParser::ConjtensorContext::getRuleIndex() const {
  return TAProLParser::RuleConjtensor;
}

void TAProLParser::ConjtensorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterConjtensor(this);
}

void TAProLParser::ConjtensorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitConjtensor(this);
}

TAProLParser::ConjtensorContext* TAProLParser::conjtensor() {
  ConjtensorContext *_localctx = _tracker.createInstance<ConjtensorContext>(_ctx, getState());
  enterRule(_localctx, 84, TAProLParser::RuleConjtensor);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(439);
    tensorname();
    setState(440);
    match(TAProLParser::T__30);
    setState(441);
    match(TAProLParser::T__4);
    setState(443);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAProLParser::ID) {
      setState(442);
      indexlist();
    }
    setState(445);
    match(TAProLParser::T__5);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TensornameContext ------------------------------------------------------------------

TAProLParser::TensornameContext::TensornameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::IdContext* TAProLParser::TensornameContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
}


size_t TAProLParser::TensornameContext::getRuleIndex() const {
  return TAProLParser::RuleTensorname;
}

void TAProLParser::TensornameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTensorname(this);
}

void TAProLParser::TensornameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTensorname(this);
}

TAProLParser::TensornameContext* TAProLParser::tensorname() {
  TensornameContext *_localctx = _tracker.createInstance<TensornameContext>(_ctx, getState());
  enterRule(_localctx, 86, TAProLParser::RuleTensorname);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(447);
    id();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IdContext ------------------------------------------------------------------

TAProLParser::IdContext::IdContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAProLParser::IdContext::ID() {
  return getToken(TAProLParser::ID, 0);
}


size_t TAProLParser::IdContext::getRuleIndex() const {
  return TAProLParser::RuleId;
}

void TAProLParser::IdContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterId(this);
}

void TAProLParser::IdContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitId(this);
}

TAProLParser::IdContext* TAProLParser::id() {
  IdContext *_localctx = _tracker.createInstance<IdContext>(_ctx, getState());
  enterRule(_localctx, 88, TAProLParser::RuleId);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(449);
    match(TAProLParser::ID);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ComplexContext ------------------------------------------------------------------

TAProLParser::ComplexContext::ComplexContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAProLParser::RealContext *> TAProLParser::ComplexContext::real() {
  return getRuleContexts<TAProLParser::RealContext>();
}

TAProLParser::RealContext* TAProLParser::ComplexContext::real(size_t i) {
  return getRuleContext<TAProLParser::RealContext>(i);
}


size_t TAProLParser::ComplexContext::getRuleIndex() const {
  return TAProLParser::RuleComplex;
}

void TAProLParser::ComplexContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterComplex(this);
}

void TAProLParser::ComplexContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitComplex(this);
}

TAProLParser::ComplexContext* TAProLParser::complex() {
  ComplexContext *_localctx = _tracker.createInstance<ComplexContext>(_ctx, getState());
  enterRule(_localctx, 90, TAProLParser::RuleComplex);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(451);
    match(TAProLParser::T__31);
    setState(452);
    real();
    setState(453);
    match(TAProLParser::T__7);
    setState(454);
    real();
    setState(455);
    match(TAProLParser::T__32);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RealContext ------------------------------------------------------------------

TAProLParser::RealContext::RealContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAProLParser::RealContext::REAL() {
  return getToken(TAProLParser::REAL, 0);
}

tree::TerminalNode* TAProLParser::RealContext::FLOAT() {
  return getToken(TAProLParser::FLOAT, 0);
}


size_t TAProLParser::RealContext::getRuleIndex() const {
  return TAProLParser::RuleReal;
}

void TAProLParser::RealContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReal(this);
}

void TAProLParser::RealContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReal(this);
}

TAProLParser::RealContext* TAProLParser::real() {
  RealContext *_localctx = _tracker.createInstance<RealContext>(_ctx, getState());
  enterRule(_localctx, 92, TAProLParser::RuleReal);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(457);
    _la = _input->LA(1);
    if (!(_la == TAProLParser::REAL

    || _la == TAProLParser::FLOAT)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StringContext ------------------------------------------------------------------

TAProLParser::StringContext::StringContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAProLParser::StringContext::STRING() {
  return getToken(TAProLParser::STRING, 0);
}


size_t TAProLParser::StringContext::getRuleIndex() const {
  return TAProLParser::RuleString;
}

void TAProLParser::StringContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterString(this);
}

void TAProLParser::StringContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitString(this);
}

TAProLParser::StringContext* TAProLParser::string() {
  StringContext *_localctx = _tracker.createInstance<StringContext>(_ctx, getState());
  enterRule(_localctx, 94, TAProLParser::RuleString);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(459);
    match(TAProLParser::STRING);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CommentContext ------------------------------------------------------------------

TAProLParser::CommentContext::CommentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAProLParser::CommentContext::COMMENT() {
  return getToken(TAProLParser::COMMENT, 0);
}


size_t TAProLParser::CommentContext::getRuleIndex() const {
  return TAProLParser::RuleComment;
}

void TAProLParser::CommentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterComment(this);
}

void TAProLParser::CommentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitComment(this);
}

TAProLParser::CommentContext* TAProLParser::comment() {
  CommentContext *_localctx = _tracker.createInstance<CommentContext>(_ctx, getState());
  enterRule(_localctx, 96, TAProLParser::RuleComment);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(461);
    match(TAProLParser::COMMENT);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

// Static vars and initialization.
std::vector<dfa::DFA> TAProLParser::_decisionToDFA;
atn::PredictionContextCache TAProLParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN TAProLParser::_atn;
std::vector<uint16_t> TAProLParser::_serializedATN;

std::vector<std::string> TAProLParser::_ruleNames = {
  "taprolsrc", "entry", "scope", "scopename", "groupnamelist", "groupname", 
  "code", "line", "statement", "simpleop", "compositeop", "space", "numfield", 
  "subspace", "spacedeflist", "spacedef", "spacename", "range", "lowerbound", 
  "upperbound", "index", "indexlist", "indexname", "assign", "datacontainer", 
  "methodname", "retrieve", "load", "save", "tagname", "destroy", "tensorlist", 
  "norm", "scalar", "scale", "prefactor", "copy", "addition", "contraction", 
  "compositeproduct", "tensornetwork", "tensor", "conjtensor", "tensorname", 
  "id", "complex", "real", "string", "comment"
};

std::vector<std::string> TAProLParser::_literalNames = {
  "", "'entry'", "':'", "'scope'", "'group'", "'('", "')'", "'end scope'", 
  "','", "'space'", "'real'", "'complex'", "'subspace'", "'='", "'['", "']'", 
  "'index'", "'?'", "'method'", "'=>'", "'load'", "'tag'", "'save'", "'~'", 
  "'destroy'", "'norm1'", "'norm2'", "'maxabs'", "'*='", "'+='", "'*'", 
  "'+'", "'{'", "'}'"
};

std::vector<std::string> TAProLParser::_symbolicNames = {
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "COMMENT", 
  "ID", "REAL", "FLOAT", "INT", "ZINT", "STRING", "WS", "EOL"
};

dfa::Vocabulary TAProLParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> TAProLParser::_tokenNames;

TAProLParser::Initializer::Initializer() {
	for (size_t i = 0; i < _symbolicNames.size(); ++i) {
		std::string name = _vocabulary.getLiteralName(i);
		if (name.empty()) {
			name = _vocabulary.getSymbolicName(i);
		}

		if (name.empty()) {
			_tokenNames.push_back("<INVALID>");
		} else {
      _tokenNames.push_back(name);
    }
	}

  _serializedATN = {
    0x3, 0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 
    0x3, 0x2c, 0x1d2, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
    0x9, 0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 
    0x4, 0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 
    0x9, 0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 
    0x4, 0xf, 0x9, 0xf, 0x4, 0x10, 0x9, 0x10, 0x4, 0x11, 0x9, 0x11, 0x4, 
    0x12, 0x9, 0x12, 0x4, 0x13, 0x9, 0x13, 0x4, 0x14, 0x9, 0x14, 0x4, 0x15, 
    0x9, 0x15, 0x4, 0x16, 0x9, 0x16, 0x4, 0x17, 0x9, 0x17, 0x4, 0x18, 0x9, 
    0x18, 0x4, 0x19, 0x9, 0x19, 0x4, 0x1a, 0x9, 0x1a, 0x4, 0x1b, 0x9, 0x1b, 
    0x4, 0x1c, 0x9, 0x1c, 0x4, 0x1d, 0x9, 0x1d, 0x4, 0x1e, 0x9, 0x1e, 0x4, 
    0x1f, 0x9, 0x1f, 0x4, 0x20, 0x9, 0x20, 0x4, 0x21, 0x9, 0x21, 0x4, 0x22, 
    0x9, 0x22, 0x4, 0x23, 0x9, 0x23, 0x4, 0x24, 0x9, 0x24, 0x4, 0x25, 0x9, 
    0x25, 0x4, 0x26, 0x9, 0x26, 0x4, 0x27, 0x9, 0x27, 0x4, 0x28, 0x9, 0x28, 
    0x4, 0x29, 0x9, 0x29, 0x4, 0x2a, 0x9, 0x2a, 0x4, 0x2b, 0x9, 0x2b, 0x4, 
    0x2c, 0x9, 0x2c, 0x4, 0x2d, 0x9, 0x2d, 0x4, 0x2e, 0x9, 0x2e, 0x4, 0x2f, 
    0x9, 0x2f, 0x4, 0x30, 0x9, 0x30, 0x4, 0x31, 0x9, 0x31, 0x4, 0x32, 0x9, 
    0x32, 0x3, 0x2, 0x3, 0x2, 0x6, 0x2, 0x67, 0xa, 0x2, 0xd, 0x2, 0xe, 0x2, 
    0x68, 0x3, 0x2, 0x5, 0x2, 0x6c, 0xa, 0x2, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x5, 0x4, 
    0x77, 0xa, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 
    0x5, 0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x7, 0x6, 0x83, 0xa, 0x6, 
    0xc, 0x6, 0xe, 0x6, 0x86, 0xb, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 0x8, 0x7, 
    0x8, 0x8b, 0xa, 0x8, 0xc, 0x8, 0xe, 0x8, 0x8e, 0xb, 0x8, 0x3, 0x9, 0x3, 
    0x9, 0x5, 0x9, 0x92, 0xa, 0x9, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 
    0x3, 0xa, 0x5, 0xa, 0x99, 0xa, 0xa, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 
    0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 
    0xb, 0xa5, 0xa, 0xb, 0x3, 0xc, 0x3, 0xc, 0x5, 0xc, 0xa9, 0xa, 0xc, 0x3, 
    0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 
    0xe, 0x3, 0xe, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x5, 0xf, 0xb7, 0xa, 0xf, 
    0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 
    0x7, 0x10, 0xc0, 0xa, 0x10, 0xc, 0x10, 0xe, 0x10, 0xc3, 0xb, 0x10, 0x3, 
    0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x13, 
    0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x14, 0x3, 
    0x14, 0x5, 0x14, 0xd3, 0xa, 0x14, 0x3, 0x15, 0x3, 0x15, 0x5, 0x15, 0xd7, 
    0xa, 0x15, 0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x3, 
    0x16, 0x3, 0x16, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x7, 0x17, 0xe3, 0xa, 
    0x17, 0xc, 0x17, 0xe, 0x17, 0xe6, 0xb, 0x17, 0x3, 0x18, 0x3, 0x18, 0x3, 
    0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 
    0x3, 0x19, 0x5, 0x19, 0xf2, 0xa, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 
    0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 
    0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 
    0x3, 0x19, 0x3, 0x19, 0x5, 0x19, 0x106, 0xa, 0x19, 0x3, 0x1a, 0x3, 0x1a, 
    0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 
    0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x5, 0x1c, 0x114, 0xa, 0x1c, 
    0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 0x119, 0xa, 0x1d, 0x3, 0x1d, 
    0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1e, 0x3, 
    0x1e, 0x3, 0x1e, 0x5, 0x1e, 0x124, 0xa, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 
    0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1f, 0x3, 0x1f, 0x3, 
    0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x5, 0x20, 
    0x134, 0xa, 0x20, 0x3, 0x21, 0x3, 0x21, 0x5, 0x21, 0x138, 0xa, 0x21, 
    0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x5, 0x21, 0x13d, 0xa, 0x21, 0x7, 0x21, 
    0x13f, 0xa, 0x21, 0xc, 0x21, 0xe, 0x21, 0x142, 0xb, 0x21, 0x3, 0x22, 
    0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x5, 0x22, 0x14a, 
    0xa, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 
    0x22, 0x3, 0x22, 0x3, 0x22, 0x5, 0x22, 0x154, 0xa, 0x22, 0x3, 0x22, 
    0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 
    0x22, 0x5, 0x22, 0x15e, 0xa, 0x22, 0x3, 0x22, 0x3, 0x22, 0x5, 0x22, 
    0x162, 0xa, 0x22, 0x3, 0x23, 0x3, 0x23, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 
    0x3, 0x24, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x5, 0x25, 0x16d, 0xa, 0x25, 
    0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 0x27, 0x3, 0x27, 0x3, 
    0x27, 0x3, 0x27, 0x5, 0x27, 0x177, 0xa, 0x27, 0x3, 0x27, 0x3, 0x27, 
    0x5, 0x27, 0x17b, 0xa, 0x27, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 
    0x5, 0x28, 0x181, 0xa, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x5, 0x28, 
    0x186, 0xa, 0x28, 0x3, 0x28, 0x3, 0x28, 0x5, 0x28, 0x18a, 0xa, 0x28, 
    0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x5, 0x29, 0x190, 0xa, 0x29, 
    0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x5, 0x29, 0x195, 0xa, 0x29, 0x3, 0x29, 
    0x3, 0x29, 0x3, 0x29, 0x5, 0x29, 0x19a, 0xa, 0x29, 0x6, 0x29, 0x19c, 
    0xa, 0x29, 0xd, 0x29, 0xe, 0x29, 0x19d, 0x3, 0x29, 0x3, 0x29, 0x5, 0x29, 
    0x1a2, 0xa, 0x29, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x5, 0x2a, 
    0x1a8, 0xa, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x5, 0x2a, 0x1ad, 
    0xa, 0x2a, 0x6, 0x2a, 0x1af, 0xa, 0x2a, 0xd, 0x2a, 0xe, 0x2a, 0x1b0, 
    0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x5, 0x2b, 0x1b6, 0xa, 0x2b, 0x3, 0x2b, 
    0x3, 0x2b, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x5, 0x2c, 0x1be, 
    0xa, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2e, 0x3, 
    0x2e, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 
    0x3, 0x30, 0x3, 0x30, 0x3, 0x31, 0x3, 0x31, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x2, 0x2, 0x33, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 
    0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 
    0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e, 0x40, 0x42, 
    0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x5a, 
    0x5c, 0x5e, 0x60, 0x62, 0x2, 0x4, 0x3, 0x2, 0xc, 0xd, 0x3, 0x2, 0x26, 
    0x27, 0x2, 0x1dc, 0x2, 0x6b, 0x3, 0x2, 0x2, 0x2, 0x4, 0x6d, 0x3, 0x2, 
    0x2, 0x2, 0x6, 0x71, 0x3, 0x2, 0x2, 0x2, 0x8, 0x7d, 0x3, 0x2, 0x2, 0x2, 
    0xa, 0x7f, 0x3, 0x2, 0x2, 0x2, 0xc, 0x87, 0x3, 0x2, 0x2, 0x2, 0xe, 0x8c, 
    0x3, 0x2, 0x2, 0x2, 0x10, 0x91, 0x3, 0x2, 0x2, 0x2, 0x12, 0x98, 0x3, 
    0x2, 0x2, 0x2, 0x14, 0xa4, 0x3, 0x2, 0x2, 0x2, 0x16, 0xa8, 0x3, 0x2, 
    0x2, 0x2, 0x18, 0xaa, 0x3, 0x2, 0x2, 0x2, 0x1a, 0xb1, 0x3, 0x2, 0x2, 
    0x2, 0x1c, 0xb3, 0x3, 0x2, 0x2, 0x2, 0x1e, 0xbc, 0x3, 0x2, 0x2, 0x2, 
    0x20, 0xc4, 0x3, 0x2, 0x2, 0x2, 0x22, 0xc8, 0x3, 0x2, 0x2, 0x2, 0x24, 
    0xca, 0x3, 0x2, 0x2, 0x2, 0x26, 0xd2, 0x3, 0x2, 0x2, 0x2, 0x28, 0xd6, 
    0x3, 0x2, 0x2, 0x2, 0x2a, 0xd8, 0x3, 0x2, 0x2, 0x2, 0x2c, 0xdf, 0x3, 
    0x2, 0x2, 0x2, 0x2e, 0xe7, 0x3, 0x2, 0x2, 0x2, 0x30, 0x105, 0x3, 0x2, 
    0x2, 0x2, 0x32, 0x107, 0x3, 0x2, 0x2, 0x2, 0x34, 0x109, 0x3, 0x2, 0x2, 
    0x2, 0x36, 0x113, 0x3, 0x2, 0x2, 0x2, 0x38, 0x115, 0x3, 0x2, 0x2, 0x2, 
    0x3a, 0x120, 0x3, 0x2, 0x2, 0x2, 0x3c, 0x12b, 0x3, 0x2, 0x2, 0x2, 0x3e, 
    0x133, 0x3, 0x2, 0x2, 0x2, 0x40, 0x137, 0x3, 0x2, 0x2, 0x2, 0x42, 0x161, 
    0x3, 0x2, 0x2, 0x2, 0x44, 0x163, 0x3, 0x2, 0x2, 0x2, 0x46, 0x165, 0x3, 
    0x2, 0x2, 0x2, 0x48, 0x16c, 0x3, 0x2, 0x2, 0x2, 0x4a, 0x16e, 0x3, 0x2, 
    0x2, 0x2, 0x4c, 0x172, 0x3, 0x2, 0x2, 0x2, 0x4e, 0x17c, 0x3, 0x2, 0x2, 
    0x2, 0x50, 0x18b, 0x3, 0x2, 0x2, 0x2, 0x52, 0x1a3, 0x3, 0x2, 0x2, 0x2, 
    0x54, 0x1b2, 0x3, 0x2, 0x2, 0x2, 0x56, 0x1b9, 0x3, 0x2, 0x2, 0x2, 0x58, 
    0x1c1, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x1c3, 0x3, 0x2, 0x2, 0x2, 0x5c, 0x1c5, 
    0x3, 0x2, 0x2, 0x2, 0x5e, 0x1cb, 0x3, 0x2, 0x2, 0x2, 0x60, 0x1cd, 0x3, 
    0x2, 0x2, 0x2, 0x62, 0x1cf, 0x3, 0x2, 0x2, 0x2, 0x64, 0x66, 0x5, 0x4, 
    0x3, 0x2, 0x65, 0x67, 0x5, 0x6, 0x4, 0x2, 0x66, 0x65, 0x3, 0x2, 0x2, 
    0x2, 0x67, 0x68, 0x3, 0x2, 0x2, 0x2, 0x68, 0x66, 0x3, 0x2, 0x2, 0x2, 
    0x68, 0x69, 0x3, 0x2, 0x2, 0x2, 0x69, 0x6c, 0x3, 0x2, 0x2, 0x2, 0x6a, 
    0x6c, 0x5, 0xe, 0x8, 0x2, 0x6b, 0x64, 0x3, 0x2, 0x2, 0x2, 0x6b, 0x6a, 
    0x3, 0x2, 0x2, 0x2, 0x6c, 0x3, 0x3, 0x2, 0x2, 0x2, 0x6d, 0x6e, 0x7, 
    0x3, 0x2, 0x2, 0x6e, 0x6f, 0x7, 0x4, 0x2, 0x2, 0x6f, 0x70, 0x5, 0x8, 
    0x5, 0x2, 0x70, 0x5, 0x3, 0x2, 0x2, 0x2, 0x71, 0x72, 0x7, 0x5, 0x2, 
    0x2, 0x72, 0x73, 0x5, 0x8, 0x5, 0x2, 0x73, 0x74, 0x7, 0x6, 0x2, 0x2, 
    0x74, 0x76, 0x7, 0x7, 0x2, 0x2, 0x75, 0x77, 0x5, 0xa, 0x6, 0x2, 0x76, 
    0x75, 0x3, 0x2, 0x2, 0x2, 0x76, 0x77, 0x3, 0x2, 0x2, 0x2, 0x77, 0x78, 
    0x3, 0x2, 0x2, 0x2, 0x78, 0x79, 0x7, 0x8, 0x2, 0x2, 0x79, 0x7a, 0x5, 
    0xe, 0x8, 0x2, 0x7a, 0x7b, 0x7, 0x9, 0x2, 0x2, 0x7b, 0x7c, 0x5, 0x8, 
    0x5, 0x2, 0x7c, 0x7, 0x3, 0x2, 0x2, 0x2, 0x7d, 0x7e, 0x5, 0x5a, 0x2e, 
    0x2, 0x7e, 0x9, 0x3, 0x2, 0x2, 0x2, 0x7f, 0x84, 0x5, 0xc, 0x7, 0x2, 
    0x80, 0x81, 0x7, 0xa, 0x2, 0x2, 0x81, 0x83, 0x5, 0xc, 0x7, 0x2, 0x82, 
    0x80, 0x3, 0x2, 0x2, 0x2, 0x83, 0x86, 0x3, 0x2, 0x2, 0x2, 0x84, 0x82, 
    0x3, 0x2, 0x2, 0x2, 0x84, 0x85, 0x3, 0x2, 0x2, 0x2, 0x85, 0xb, 0x3, 
    0x2, 0x2, 0x2, 0x86, 0x84, 0x3, 0x2, 0x2, 0x2, 0x87, 0x88, 0x5, 0x5a, 
    0x2e, 0x2, 0x88, 0xd, 0x3, 0x2, 0x2, 0x2, 0x89, 0x8b, 0x5, 0x10, 0x9, 
    0x2, 0x8a, 0x89, 0x3, 0x2, 0x2, 0x2, 0x8b, 0x8e, 0x3, 0x2, 0x2, 0x2, 
    0x8c, 0x8a, 0x3, 0x2, 0x2, 0x2, 0x8c, 0x8d, 0x3, 0x2, 0x2, 0x2, 0x8d, 
    0xf, 0x3, 0x2, 0x2, 0x2, 0x8e, 0x8c, 0x3, 0x2, 0x2, 0x2, 0x8f, 0x92, 
    0x5, 0x12, 0xa, 0x2, 0x90, 0x92, 0x5, 0x62, 0x32, 0x2, 0x91, 0x8f, 0x3, 
    0x2, 0x2, 0x2, 0x91, 0x90, 0x3, 0x2, 0x2, 0x2, 0x92, 0x11, 0x3, 0x2, 
    0x2, 0x2, 0x93, 0x99, 0x5, 0x18, 0xd, 0x2, 0x94, 0x99, 0x5, 0x1c, 0xf, 
    0x2, 0x95, 0x99, 0x5, 0x2a, 0x16, 0x2, 0x96, 0x99, 0x5, 0x14, 0xb, 0x2, 
    0x97, 0x99, 0x5, 0x16, 0xc, 0x2, 0x98, 0x93, 0x3, 0x2, 0x2, 0x2, 0x98, 
    0x94, 0x3, 0x2, 0x2, 0x2, 0x98, 0x95, 0x3, 0x2, 0x2, 0x2, 0x98, 0x96, 
    0x3, 0x2, 0x2, 0x2, 0x98, 0x97, 0x3, 0x2, 0x2, 0x2, 0x99, 0x13, 0x3, 
    0x2, 0x2, 0x2, 0x9a, 0xa5, 0x5, 0x30, 0x19, 0x2, 0x9b, 0xa5, 0x5, 0x36, 
    0x1c, 0x2, 0x9c, 0xa5, 0x5, 0x38, 0x1d, 0x2, 0x9d, 0xa5, 0x5, 0x3a, 
    0x1e, 0x2, 0x9e, 0xa5, 0x5, 0x3e, 0x20, 0x2, 0x9f, 0xa5, 0x5, 0x42, 
    0x22, 0x2, 0xa0, 0xa5, 0x5, 0x46, 0x24, 0x2, 0xa1, 0xa5, 0x5, 0x4a, 
    0x26, 0x2, 0xa2, 0xa5, 0x5, 0x4c, 0x27, 0x2, 0xa3, 0xa5, 0x5, 0x4e, 
    0x28, 0x2, 0xa4, 0x9a, 0x3, 0x2, 0x2, 0x2, 0xa4, 0x9b, 0x3, 0x2, 0x2, 
    0x2, 0xa4, 0x9c, 0x3, 0x2, 0x2, 0x2, 0xa4, 0x9d, 0x3, 0x2, 0x2, 0x2, 
    0xa4, 0x9e, 0x3, 0x2, 0x2, 0x2, 0xa4, 0x9f, 0x3, 0x2, 0x2, 0x2, 0xa4, 
    0xa0, 0x3, 0x2, 0x2, 0x2, 0xa4, 0xa1, 0x3, 0x2, 0x2, 0x2, 0xa4, 0xa2, 
    0x3, 0x2, 0x2, 0x2, 0xa4, 0xa3, 0x3, 0x2, 0x2, 0x2, 0xa5, 0x15, 0x3, 
    0x2, 0x2, 0x2, 0xa6, 0xa9, 0x5, 0x50, 0x29, 0x2, 0xa7, 0xa9, 0x5, 0x52, 
    0x2a, 0x2, 0xa8, 0xa6, 0x3, 0x2, 0x2, 0x2, 0xa8, 0xa7, 0x3, 0x2, 0x2, 
    0x2, 0xa9, 0x17, 0x3, 0x2, 0x2, 0x2, 0xaa, 0xab, 0x7, 0xb, 0x2, 0x2, 
    0xab, 0xac, 0x7, 0x7, 0x2, 0x2, 0xac, 0xad, 0x5, 0x1a, 0xe, 0x2, 0xad, 
    0xae, 0x7, 0x8, 0x2, 0x2, 0xae, 0xaf, 0x7, 0x4, 0x2, 0x2, 0xaf, 0xb0, 
    0x5, 0x1e, 0x10, 0x2, 0xb0, 0x19, 0x3, 0x2, 0x2, 0x2, 0xb1, 0xb2, 0x9, 
    0x2, 0x2, 0x2, 0xb2, 0x1b, 0x3, 0x2, 0x2, 0x2, 0xb3, 0xb4, 0x7, 0xe, 
    0x2, 0x2, 0xb4, 0xb6, 0x7, 0x7, 0x2, 0x2, 0xb5, 0xb7, 0x5, 0x22, 0x12, 
    0x2, 0xb6, 0xb5, 0x3, 0x2, 0x2, 0x2, 0xb6, 0xb7, 0x3, 0x2, 0x2, 0x2, 
    0xb7, 0xb8, 0x3, 0x2, 0x2, 0x2, 0xb8, 0xb9, 0x7, 0x8, 0x2, 0x2, 0xb9, 
    0xba, 0x7, 0x4, 0x2, 0x2, 0xba, 0xbb, 0x5, 0x1e, 0x10, 0x2, 0xbb, 0x1d, 
    0x3, 0x2, 0x2, 0x2, 0xbc, 0xc1, 0x5, 0x20, 0x11, 0x2, 0xbd, 0xbe, 0x7, 
    0xa, 0x2, 0x2, 0xbe, 0xc0, 0x5, 0x20, 0x11, 0x2, 0xbf, 0xbd, 0x3, 0x2, 
    0x2, 0x2, 0xc0, 0xc3, 0x3, 0x2, 0x2, 0x2, 0xc1, 0xbf, 0x3, 0x2, 0x2, 
    0x2, 0xc1, 0xc2, 0x3, 0x2, 0x2, 0x2, 0xc2, 0x1f, 0x3, 0x2, 0x2, 0x2, 
    0xc3, 0xc1, 0x3, 0x2, 0x2, 0x2, 0xc4, 0xc5, 0x5, 0x22, 0x12, 0x2, 0xc5, 
    0xc6, 0x7, 0xf, 0x2, 0x2, 0xc6, 0xc7, 0x5, 0x24, 0x13, 0x2, 0xc7, 0x21, 
    0x3, 0x2, 0x2, 0x2, 0xc8, 0xc9, 0x5, 0x5a, 0x2e, 0x2, 0xc9, 0x23, 0x3, 
    0x2, 0x2, 0x2, 0xca, 0xcb, 0x7, 0x10, 0x2, 0x2, 0xcb, 0xcc, 0x5, 0x26, 
    0x14, 0x2, 0xcc, 0xcd, 0x7, 0x4, 0x2, 0x2, 0xcd, 0xce, 0x5, 0x28, 0x15, 
    0x2, 0xce, 0xcf, 0x7, 0x11, 0x2, 0x2, 0xcf, 0x25, 0x3, 0x2, 0x2, 0x2, 
    0xd0, 0xd3, 0x7, 0x28, 0x2, 0x2, 0xd1, 0xd3, 0x5, 0x5a, 0x2e, 0x2, 0xd2, 
    0xd0, 0x3, 0x2, 0x2, 0x2, 0xd2, 0xd1, 0x3, 0x2, 0x2, 0x2, 0xd3, 0x27, 
    0x3, 0x2, 0x2, 0x2, 0xd4, 0xd7, 0x7, 0x28, 0x2, 0x2, 0xd5, 0xd7, 0x5, 
    0x5a, 0x2e, 0x2, 0xd6, 0xd4, 0x3, 0x2, 0x2, 0x2, 0xd6, 0xd5, 0x3, 0x2, 
    0x2, 0x2, 0xd7, 0x29, 0x3, 0x2, 0x2, 0x2, 0xd8, 0xd9, 0x7, 0x12, 0x2, 
    0x2, 0xd9, 0xda, 0x7, 0x7, 0x2, 0x2, 0xda, 0xdb, 0x5, 0x22, 0x12, 0x2, 
    0xdb, 0xdc, 0x7, 0x8, 0x2, 0x2, 0xdc, 0xdd, 0x7, 0x4, 0x2, 0x2, 0xdd, 
    0xde, 0x5, 0x2c, 0x17, 0x2, 0xde, 0x2b, 0x3, 0x2, 0x2, 0x2, 0xdf, 0xe4, 
    0x5, 0x2e, 0x18, 0x2, 0xe0, 0xe1, 0x7, 0xa, 0x2, 0x2, 0xe1, 0xe3, 0x5, 
    0x2e, 0x18, 0x2, 0xe2, 0xe0, 0x3, 0x2, 0x2, 0x2, 0xe3, 0xe6, 0x3, 0x2, 
    0x2, 0x2, 0xe4, 0xe2, 0x3, 0x2, 0x2, 0x2, 0xe4, 0xe5, 0x3, 0x2, 0x2, 
    0x2, 0xe5, 0x2d, 0x3, 0x2, 0x2, 0x2, 0xe6, 0xe4, 0x3, 0x2, 0x2, 0x2, 
    0xe7, 0xe8, 0x5, 0x5a, 0x2e, 0x2, 0xe8, 0x2f, 0x3, 0x2, 0x2, 0x2, 0xe9, 
    0xea, 0x5, 0x54, 0x2b, 0x2, 0xea, 0xeb, 0x7, 0xf, 0x2, 0x2, 0xeb, 0xec, 
    0x7, 0x13, 0x2, 0x2, 0xec, 0x106, 0x3, 0x2, 0x2, 0x2, 0xed, 0xee, 0x5, 
    0x54, 0x2b, 0x2, 0xee, 0xf1, 0x7, 0xf, 0x2, 0x2, 0xef, 0xf2, 0x5, 0x5e, 
    0x30, 0x2, 0xf0, 0xf2, 0x5, 0x5c, 0x2f, 0x2, 0xf1, 0xef, 0x3, 0x2, 0x2, 
    0x2, 0xf1, 0xf0, 0x3, 0x2, 0x2, 0x2, 0xf2, 0x106, 0x3, 0x2, 0x2, 0x2, 
    0xf3, 0xf4, 0x5, 0x54, 0x2b, 0x2, 0xf4, 0xf5, 0x7, 0xf, 0x2, 0x2, 0xf5, 
    0xf6, 0x5, 0x32, 0x1a, 0x2, 0xf6, 0x106, 0x3, 0x2, 0x2, 0x2, 0xf7, 0xf8, 
    0x5, 0x54, 0x2b, 0x2, 0xf8, 0xf9, 0x7, 0xf, 0x2, 0x2, 0xf9, 0xfa, 0x7, 
    0x14, 0x2, 0x2, 0xfa, 0xfb, 0x7, 0x7, 0x2, 0x2, 0xfb, 0xfc, 0x5, 0x34, 
    0x1b, 0x2, 0xfc, 0xfd, 0x7, 0x8, 0x2, 0x2, 0xfd, 0x106, 0x3, 0x2, 0x2, 
    0x2, 0xfe, 0xff, 0x5, 0x54, 0x2b, 0x2, 0xff, 0x100, 0x7, 0x15, 0x2, 
    0x2, 0x100, 0x101, 0x7, 0x14, 0x2, 0x2, 0x101, 0x102, 0x7, 0x7, 0x2, 
    0x2, 0x102, 0x103, 0x5, 0x34, 0x1b, 0x2, 0x103, 0x104, 0x7, 0x8, 0x2, 
    0x2, 0x104, 0x106, 0x3, 0x2, 0x2, 0x2, 0x105, 0xe9, 0x3, 0x2, 0x2, 0x2, 
    0x105, 0xed, 0x3, 0x2, 0x2, 0x2, 0x105, 0xf3, 0x3, 0x2, 0x2, 0x2, 0x105, 
    0xf7, 0x3, 0x2, 0x2, 0x2, 0x105, 0xfe, 0x3, 0x2, 0x2, 0x2, 0x106, 0x31, 
    0x3, 0x2, 0x2, 0x2, 0x107, 0x108, 0x5, 0x5a, 0x2e, 0x2, 0x108, 0x33, 
    0x3, 0x2, 0x2, 0x2, 0x109, 0x10a, 0x5, 0x60, 0x31, 0x2, 0x10a, 0x35, 
    0x3, 0x2, 0x2, 0x2, 0x10b, 0x10c, 0x5, 0x32, 0x1a, 0x2, 0x10c, 0x10d, 
    0x7, 0xf, 0x2, 0x2, 0x10d, 0x10e, 0x5, 0x58, 0x2d, 0x2, 0x10e, 0x114, 
    0x3, 0x2, 0x2, 0x2, 0x10f, 0x110, 0x5, 0x32, 0x1a, 0x2, 0x110, 0x111, 
    0x7, 0xf, 0x2, 0x2, 0x111, 0x112, 0x5, 0x54, 0x2b, 0x2, 0x112, 0x114, 
    0x3, 0x2, 0x2, 0x2, 0x113, 0x10b, 0x3, 0x2, 0x2, 0x2, 0x113, 0x10f, 
    0x3, 0x2, 0x2, 0x2, 0x114, 0x37, 0x3, 0x2, 0x2, 0x2, 0x115, 0x118, 0x7, 
    0x16, 0x2, 0x2, 0x116, 0x119, 0x5, 0x54, 0x2b, 0x2, 0x117, 0x119, 0x5, 
    0x58, 0x2d, 0x2, 0x118, 0x116, 0x3, 0x2, 0x2, 0x2, 0x118, 0x117, 0x3, 
    0x2, 0x2, 0x2, 0x119, 0x11a, 0x3, 0x2, 0x2, 0x2, 0x11a, 0x11b, 0x7, 
    0x4, 0x2, 0x2, 0x11b, 0x11c, 0x7, 0x17, 0x2, 0x2, 0x11c, 0x11d, 0x7, 
    0x7, 0x2, 0x2, 0x11d, 0x11e, 0x5, 0x3c, 0x1f, 0x2, 0x11e, 0x11f, 0x7, 
    0x8, 0x2, 0x2, 0x11f, 0x39, 0x3, 0x2, 0x2, 0x2, 0x120, 0x123, 0x7, 0x18, 
    0x2, 0x2, 0x121, 0x124, 0x5, 0x54, 0x2b, 0x2, 0x122, 0x124, 0x5, 0x58, 
    0x2d, 0x2, 0x123, 0x121, 0x3, 0x2, 0x2, 0x2, 0x123, 0x122, 0x3, 0x2, 
    0x2, 0x2, 0x124, 0x125, 0x3, 0x2, 0x2, 0x2, 0x125, 0x126, 0x7, 0x4, 
    0x2, 0x2, 0x126, 0x127, 0x7, 0x17, 0x2, 0x2, 0x127, 0x128, 0x7, 0x7, 
    0x2, 0x2, 0x128, 0x129, 0x5, 0x3c, 0x1f, 0x2, 0x129, 0x12a, 0x7, 0x8, 
    0x2, 0x2, 0x12a, 0x3b, 0x3, 0x2, 0x2, 0x2, 0x12b, 0x12c, 0x5, 0x60, 
    0x31, 0x2, 0x12c, 0x3d, 0x3, 0x2, 0x2, 0x2, 0x12d, 0x12e, 0x7, 0x19, 
    0x2, 0x2, 0x12e, 0x134, 0x5, 0x58, 0x2d, 0x2, 0x12f, 0x130, 0x7, 0x19, 
    0x2, 0x2, 0x130, 0x134, 0x5, 0x54, 0x2b, 0x2, 0x131, 0x132, 0x7, 0x1a, 
    0x2, 0x2, 0x132, 0x134, 0x5, 0x40, 0x21, 0x2, 0x133, 0x12d, 0x3, 0x2, 
    0x2, 0x2, 0x133, 0x12f, 0x3, 0x2, 0x2, 0x2, 0x133, 0x131, 0x3, 0x2, 
    0x2, 0x2, 0x134, 0x3f, 0x3, 0x2, 0x2, 0x2, 0x135, 0x138, 0x5, 0x58, 
    0x2d, 0x2, 0x136, 0x138, 0x5, 0x54, 0x2b, 0x2, 0x137, 0x135, 0x3, 0x2, 
    0x2, 0x2, 0x137, 0x136, 0x3, 0x2, 0x2, 0x2, 0x138, 0x140, 0x3, 0x2, 
    0x2, 0x2, 0x139, 0x13c, 0x7, 0xa, 0x2, 0x2, 0x13a, 0x13d, 0x5, 0x58, 
    0x2d, 0x2, 0x13b, 0x13d, 0x5, 0x54, 0x2b, 0x2, 0x13c, 0x13a, 0x3, 0x2, 
    0x2, 0x2, 0x13c, 0x13b, 0x3, 0x2, 0x2, 0x2, 0x13d, 0x13f, 0x3, 0x2, 
    0x2, 0x2, 0x13e, 0x139, 0x3, 0x2, 0x2, 0x2, 0x13f, 0x142, 0x3, 0x2, 
    0x2, 0x2, 0x140, 0x13e, 0x3, 0x2, 0x2, 0x2, 0x140, 0x141, 0x3, 0x2, 
    0x2, 0x2, 0x141, 0x41, 0x3, 0x2, 0x2, 0x2, 0x142, 0x140, 0x3, 0x2, 0x2, 
    0x2, 0x143, 0x144, 0x5, 0x44, 0x23, 0x2, 0x144, 0x145, 0x7, 0xf, 0x2, 
    0x2, 0x145, 0x146, 0x7, 0x1b, 0x2, 0x2, 0x146, 0x149, 0x7, 0x7, 0x2, 
    0x2, 0x147, 0x14a, 0x5, 0x58, 0x2d, 0x2, 0x148, 0x14a, 0x5, 0x54, 0x2b, 
    0x2, 0x149, 0x147, 0x3, 0x2, 0x2, 0x2, 0x149, 0x148, 0x3, 0x2, 0x2, 
    0x2, 0x14a, 0x14b, 0x3, 0x2, 0x2, 0x2, 0x14b, 0x14c, 0x7, 0x8, 0x2, 
    0x2, 0x14c, 0x162, 0x3, 0x2, 0x2, 0x2, 0x14d, 0x14e, 0x5, 0x44, 0x23, 
    0x2, 0x14e, 0x14f, 0x7, 0xf, 0x2, 0x2, 0x14f, 0x150, 0x7, 0x1c, 0x2, 
    0x2, 0x150, 0x153, 0x7, 0x7, 0x2, 0x2, 0x151, 0x154, 0x5, 0x58, 0x2d, 
    0x2, 0x152, 0x154, 0x5, 0x54, 0x2b, 0x2, 0x153, 0x151, 0x3, 0x2, 0x2, 
    0x2, 0x153, 0x152, 0x3, 0x2, 0x2, 0x2, 0x154, 0x155, 0x3, 0x2, 0x2, 
    0x2, 0x155, 0x156, 0x7, 0x8, 0x2, 0x2, 0x156, 0x162, 0x3, 0x2, 0x2, 
    0x2, 0x157, 0x158, 0x5, 0x44, 0x23, 0x2, 0x158, 0x159, 0x7, 0xf, 0x2, 
    0x2, 0x159, 0x15a, 0x7, 0x1d, 0x2, 0x2, 0x15a, 0x15d, 0x7, 0x7, 0x2, 
    0x2, 0x15b, 0x15e, 0x5, 0x58, 0x2d, 0x2, 0x15c, 0x15e, 0x5, 0x54, 0x2b, 
    0x2, 0x15d, 0x15b, 0x3, 0x2, 0x2, 0x2, 0x15d, 0x15c, 0x3, 0x2, 0x2, 
    0x2, 0x15e, 0x15f, 0x3, 0x2, 0x2, 0x2, 0x15f, 0x160, 0x7, 0x8, 0x2, 
    0x2, 0x160, 0x162, 0x3, 0x2, 0x2, 0x2, 0x161, 0x143, 0x3, 0x2, 0x2, 
    0x2, 0x161, 0x14d, 0x3, 0x2, 0x2, 0x2, 0x161, 0x157, 0x3, 0x2, 0x2, 
    0x2, 0x162, 0x43, 0x3, 0x2, 0x2, 0x2, 0x163, 0x164, 0x5, 0x5a, 0x2e, 
    0x2, 0x164, 0x45, 0x3, 0x2, 0x2, 0x2, 0x165, 0x166, 0x5, 0x54, 0x2b, 
    0x2, 0x166, 0x167, 0x7, 0x1e, 0x2, 0x2, 0x167, 0x168, 0x5, 0x48, 0x25, 
    0x2, 0x168, 0x47, 0x3, 0x2, 0x2, 0x2, 0x169, 0x16d, 0x5, 0x5e, 0x30, 
    0x2, 0x16a, 0x16d, 0x5, 0x5c, 0x2f, 0x2, 0x16b, 0x16d, 0x5, 0x5a, 0x2e, 
    0x2, 0x16c, 0x169, 0x3, 0x2, 0x2, 0x2, 0x16c, 0x16a, 0x3, 0x2, 0x2, 
    0x2, 0x16c, 0x16b, 0x3, 0x2, 0x2, 0x2, 0x16d, 0x49, 0x3, 0x2, 0x2, 0x2, 
    0x16e, 0x16f, 0x5, 0x54, 0x2b, 0x2, 0x16f, 0x170, 0x7, 0xf, 0x2, 0x2, 
    0x170, 0x171, 0x5, 0x54, 0x2b, 0x2, 0x171, 0x4b, 0x3, 0x2, 0x2, 0x2, 
    0x172, 0x173, 0x5, 0x54, 0x2b, 0x2, 0x173, 0x176, 0x7, 0x1f, 0x2, 0x2, 
    0x174, 0x177, 0x5, 0x54, 0x2b, 0x2, 0x175, 0x177, 0x5, 0x56, 0x2c, 0x2, 
    0x176, 0x174, 0x3, 0x2, 0x2, 0x2, 0x176, 0x175, 0x3, 0x2, 0x2, 0x2, 
    0x177, 0x17a, 0x3, 0x2, 0x2, 0x2, 0x178, 0x179, 0x7, 0x20, 0x2, 0x2, 
    0x179, 0x17b, 0x5, 0x48, 0x25, 0x2, 0x17a, 0x178, 0x3, 0x2, 0x2, 0x2, 
    0x17a, 0x17b, 0x3, 0x2, 0x2, 0x2, 0x17b, 0x4d, 0x3, 0x2, 0x2, 0x2, 0x17c, 
    0x17d, 0x5, 0x54, 0x2b, 0x2, 0x17d, 0x180, 0x7, 0x1f, 0x2, 0x2, 0x17e, 
    0x181, 0x5, 0x54, 0x2b, 0x2, 0x17f, 0x181, 0x5, 0x56, 0x2c, 0x2, 0x180, 
    0x17e, 0x3, 0x2, 0x2, 0x2, 0x180, 0x17f, 0x3, 0x2, 0x2, 0x2, 0x181, 
    0x182, 0x3, 0x2, 0x2, 0x2, 0x182, 0x185, 0x7, 0x20, 0x2, 0x2, 0x183, 
    0x186, 0x5, 0x54, 0x2b, 0x2, 0x184, 0x186, 0x5, 0x56, 0x2c, 0x2, 0x185, 
    0x183, 0x3, 0x2, 0x2, 0x2, 0x185, 0x184, 0x3, 0x2, 0x2, 0x2, 0x186, 
    0x189, 0x3, 0x2, 0x2, 0x2, 0x187, 0x188, 0x7, 0x20, 0x2, 0x2, 0x188, 
    0x18a, 0x5, 0x48, 0x25, 0x2, 0x189, 0x187, 0x3, 0x2, 0x2, 0x2, 0x189, 
    0x18a, 0x3, 0x2, 0x2, 0x2, 0x18a, 0x4f, 0x3, 0x2, 0x2, 0x2, 0x18b, 0x18c, 
    0x5, 0x54, 0x2b, 0x2, 0x18c, 0x18f, 0x7, 0x1f, 0x2, 0x2, 0x18d, 0x190, 
    0x5, 0x54, 0x2b, 0x2, 0x18e, 0x190, 0x5, 0x56, 0x2c, 0x2, 0x18f, 0x18d, 
    0x3, 0x2, 0x2, 0x2, 0x18f, 0x18e, 0x3, 0x2, 0x2, 0x2, 0x190, 0x191, 
    0x3, 0x2, 0x2, 0x2, 0x191, 0x194, 0x7, 0x20, 0x2, 0x2, 0x192, 0x195, 
    0x5, 0x54, 0x2b, 0x2, 0x193, 0x195, 0x5, 0x56, 0x2c, 0x2, 0x194, 0x192, 
    0x3, 0x2, 0x2, 0x2, 0x194, 0x193, 0x3, 0x2, 0x2, 0x2, 0x195, 0x19b, 
    0x3, 0x2, 0x2, 0x2, 0x196, 0x199, 0x7, 0x20, 0x2, 0x2, 0x197, 0x19a, 
    0x5, 0x54, 0x2b, 0x2, 0x198, 0x19a, 0x5, 0x56, 0x2c, 0x2, 0x199, 0x197, 
    0x3, 0x2, 0x2, 0x2, 0x199, 0x198, 0x3, 0x2, 0x2, 0x2, 0x19a, 0x19c, 
    0x3, 0x2, 0x2, 0x2, 0x19b, 0x196, 0x3, 0x2, 0x2, 0x2, 0x19c, 0x19d, 
    0x3, 0x2, 0x2, 0x2, 0x19d, 0x19b, 0x3, 0x2, 0x2, 0x2, 0x19d, 0x19e, 
    0x3, 0x2, 0x2, 0x2, 0x19e, 0x1a1, 0x3, 0x2, 0x2, 0x2, 0x19f, 0x1a0, 
    0x7, 0x20, 0x2, 0x2, 0x1a0, 0x1a2, 0x5, 0x48, 0x25, 0x2, 0x1a1, 0x19f, 
    0x3, 0x2, 0x2, 0x2, 0x1a1, 0x1a2, 0x3, 0x2, 0x2, 0x2, 0x1a2, 0x51, 0x3, 
    0x2, 0x2, 0x2, 0x1a3, 0x1a4, 0x5, 0x54, 0x2b, 0x2, 0x1a4, 0x1a7, 0x7, 
    0x15, 0x2, 0x2, 0x1a5, 0x1a8, 0x5, 0x54, 0x2b, 0x2, 0x1a6, 0x1a8, 0x5, 
    0x56, 0x2c, 0x2, 0x1a7, 0x1a5, 0x3, 0x2, 0x2, 0x2, 0x1a7, 0x1a6, 0x3, 
    0x2, 0x2, 0x2, 0x1a8, 0x1ae, 0x3, 0x2, 0x2, 0x2, 0x1a9, 0x1ac, 0x7, 
    0x20, 0x2, 0x2, 0x1aa, 0x1ad, 0x5, 0x54, 0x2b, 0x2, 0x1ab, 0x1ad, 0x5, 
    0x56, 0x2c, 0x2, 0x1ac, 0x1aa, 0x3, 0x2, 0x2, 0x2, 0x1ac, 0x1ab, 0x3, 
    0x2, 0x2, 0x2, 0x1ad, 0x1af, 0x3, 0x2, 0x2, 0x2, 0x1ae, 0x1a9, 0x3, 
    0x2, 0x2, 0x2, 0x1af, 0x1b0, 0x3, 0x2, 0x2, 0x2, 0x1b0, 0x1ae, 0x3, 
    0x2, 0x2, 0x2, 0x1b0, 0x1b1, 0x3, 0x2, 0x2, 0x2, 0x1b1, 0x53, 0x3, 0x2, 
    0x2, 0x2, 0x1b2, 0x1b3, 0x5, 0x58, 0x2d, 0x2, 0x1b3, 0x1b5, 0x7, 0x7, 
    0x2, 0x2, 0x1b4, 0x1b6, 0x5, 0x2c, 0x17, 0x2, 0x1b5, 0x1b4, 0x3, 0x2, 
    0x2, 0x2, 0x1b5, 0x1b6, 0x3, 0x2, 0x2, 0x2, 0x1b6, 0x1b7, 0x3, 0x2, 
    0x2, 0x2, 0x1b7, 0x1b8, 0x7, 0x8, 0x2, 0x2, 0x1b8, 0x55, 0x3, 0x2, 0x2, 
    0x2, 0x1b9, 0x1ba, 0x5, 0x58, 0x2d, 0x2, 0x1ba, 0x1bb, 0x7, 0x21, 0x2, 
    0x2, 0x1bb, 0x1bd, 0x7, 0x7, 0x2, 0x2, 0x1bc, 0x1be, 0x5, 0x2c, 0x17, 
    0x2, 0x1bd, 0x1bc, 0x3, 0x2, 0x2, 0x2, 0x1bd, 0x1be, 0x3, 0x2, 0x2, 
    0x2, 0x1be, 0x1bf, 0x3, 0x2, 0x2, 0x2, 0x1bf, 0x1c0, 0x7, 0x8, 0x2, 
    0x2, 0x1c0, 0x57, 0x3, 0x2, 0x2, 0x2, 0x1c1, 0x1c2, 0x5, 0x5a, 0x2e, 
    0x2, 0x1c2, 0x59, 0x3, 0x2, 0x2, 0x2, 0x1c3, 0x1c4, 0x7, 0x25, 0x2, 
    0x2, 0x1c4, 0x5b, 0x3, 0x2, 0x2, 0x2, 0x1c5, 0x1c6, 0x7, 0x22, 0x2, 
    0x2, 0x1c6, 0x1c7, 0x5, 0x5e, 0x30, 0x2, 0x1c7, 0x1c8, 0x7, 0xa, 0x2, 
    0x2, 0x1c8, 0x1c9, 0x5, 0x5e, 0x30, 0x2, 0x1c9, 0x1ca, 0x7, 0x23, 0x2, 
    0x2, 0x1ca, 0x5d, 0x3, 0x2, 0x2, 0x2, 0x1cb, 0x1cc, 0x9, 0x3, 0x2, 0x2, 
    0x1cc, 0x5f, 0x3, 0x2, 0x2, 0x2, 0x1cd, 0x1ce, 0x7, 0x2a, 0x2, 0x2, 
    0x1ce, 0x61, 0x3, 0x2, 0x2, 0x2, 0x1cf, 0x1d0, 0x7, 0x24, 0x2, 0x2, 
    0x1d0, 0x63, 0x3, 0x2, 0x2, 0x2, 0x2d, 0x68, 0x6b, 0x76, 0x84, 0x8c, 
    0x91, 0x98, 0xa4, 0xa8, 0xb6, 0xc1, 0xd2, 0xd6, 0xe4, 0xf1, 0x105, 0x113, 
    0x118, 0x123, 0x133, 0x137, 0x13c, 0x140, 0x149, 0x153, 0x15d, 0x161, 
    0x16c, 0x176, 0x17a, 0x180, 0x185, 0x189, 0x18f, 0x194, 0x199, 0x19d, 
    0x1a1, 0x1a7, 0x1ac, 0x1b0, 0x1b5, 0x1bd, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

TAProLParser::Initializer TAProLParser::_init;
