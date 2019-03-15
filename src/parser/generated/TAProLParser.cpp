
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
    enterOuterAlt(_localctx, 1);
    setState(72);
    entry();
    setState(74); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(73);
      scope();
      setState(76); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == TAProLParser::T__1);
   
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

tree::TerminalNode* TAProLParser::EntryContext::ID() {
  return getToken(TAProLParser::ID, 0);
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
    setState(78);
    match(TAProLParser::T__0);
    setState(79);
    dynamic_cast<EntryContext *>(_localctx)->entryName = match(TAProLParser::ID);
   
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

TAProLParser::CodeContext* TAProLParser::ScopeContext::code() {
  return getRuleContext<TAProLParser::CodeContext>(0);
}

std::vector<TAProLParser::IdContext *> TAProLParser::ScopeContext::id() {
  return getRuleContexts<TAProLParser::IdContext>();
}

TAProLParser::IdContext* TAProLParser::ScopeContext::id(size_t i) {
  return getRuleContext<TAProLParser::IdContext>(i);
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
    setState(81);
    match(TAProLParser::T__1);
    setState(82);
    dynamic_cast<ScopeContext *>(_localctx)->scopebeginname = id();
    setState(83);
    match(TAProLParser::T__2);
    setState(84);
    match(TAProLParser::T__3);
    setState(86);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAProLParser::ID) {
      setState(85);
      groupnamelist();
    }
    setState(88);
    match(TAProLParser::T__4);
    setState(89);
    code();
    setState(90);
    match(TAProLParser::T__5);
    setState(91);
    match(TAProLParser::T__1);
    setState(92);
    dynamic_cast<ScopeContext *>(_localctx)->scopeendname = id();
   
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
  enterRule(_localctx, 6, TAProLParser::RuleCode);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(95); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(94);
      line();
      setState(97); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << TAProLParser::T__6)
      | (1ULL << TAProLParser::T__10)
      | (1ULL << TAProLParser::T__13)
      | (1ULL << TAProLParser::T__16)
      | (1ULL << TAProLParser::T__18)
      | (1ULL << TAProLParser::T__19)
      | (1ULL << TAProLParser::T__20)
      | (1ULL << TAProLParser::COMMENT)
      | (1ULL << TAProLParser::ID))) != 0));
   
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

std::vector<TAProLParser::StatementContext *> TAProLParser::LineContext::statement() {
  return getRuleContexts<TAProLParser::StatementContext>();
}

TAProLParser::StatementContext* TAProLParser::LineContext::statement(size_t i) {
  return getRuleContext<TAProLParser::StatementContext>(i);
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
  enterRule(_localctx, 8, TAProLParser::RuleLine);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    setState(105);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TAProLParser::T__6:
      case TAProLParser::T__10:
      case TAProLParser::T__13:
      case TAProLParser::T__16:
      case TAProLParser::T__18:
      case TAProLParser::T__19:
      case TAProLParser::T__20:
      case TAProLParser::ID: {
        enterOuterAlt(_localctx, 1);
        setState(100); 
        _errHandler->sync(this);
        alt = 1;
        do {
          switch (alt) {
            case 1: {
                  setState(99);
                  statement();
                  break;
                }

          default:
            throw NoViableAltException(this);
          }
          setState(102); 
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 3, _ctx);
        } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
        break;
      }

      case TAProLParser::COMMENT: {
        enterOuterAlt(_localctx, 2);
        setState(104);
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
  enterRule(_localctx, 10, TAProLParser::RuleStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(112);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(107);
      space();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(108);
      subspace();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(109);
      index();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(110);
      simpleop();
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(111);
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

TAProLParser::AssignmentContext* TAProLParser::SimpleopContext::assignment() {
  return getRuleContext<TAProLParser::AssignmentContext>(0);
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

TAProLParser::CopyContext* TAProLParser::SimpleopContext::copy() {
  return getRuleContext<TAProLParser::CopyContext>(0);
}

TAProLParser::ScaleContext* TAProLParser::SimpleopContext::scale() {
  return getRuleContext<TAProLParser::ScaleContext>(0);
}

TAProLParser::UnaryopContext* TAProLParser::SimpleopContext::unaryop() {
  return getRuleContext<TAProLParser::UnaryopContext>(0);
}

TAProLParser::BinaryopContext* TAProLParser::SimpleopContext::binaryop() {
  return getRuleContext<TAProLParser::BinaryopContext>(0);
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
  enterRule(_localctx, 12, TAProLParser::RuleSimpleop);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(122);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(114);
      assignment();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(115);
      load();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(116);
      save();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(117);
      destroy();
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(118);
      copy();
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(119);
      scale();
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(120);
      unaryop();
      break;
    }

    case 8: {
      enterOuterAlt(_localctx, 8);
      setState(121);
      binaryop();
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
  enterRule(_localctx, 14, TAProLParser::RuleCompositeop);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(126);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(124);
      compositeproduct();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(125);
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

TAProLParser::SpacelistContext* TAProLParser::SpaceContext::spacelist() {
  return getRuleContext<TAProLParser::SpacelistContext>(0);
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
  enterRule(_localctx, 16, TAProLParser::RuleSpace);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(128);
    match(TAProLParser::T__6);
    setState(129);
    match(TAProLParser::T__3);
    setState(130);
    _la = _input->LA(1);
    if (!(_la == TAProLParser::T__7

    || _la == TAProLParser::T__8)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(131);
    match(TAProLParser::T__4);
    setState(132);
    match(TAProLParser::T__9);
    setState(133);
    spacelist();
   
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

TAProLParser::SpacelistContext* TAProLParser::SubspaceContext::spacelist() {
  return getRuleContext<TAProLParser::SpacelistContext>(0);
}

TAProLParser::IdContext* TAProLParser::SubspaceContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
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
  enterRule(_localctx, 18, TAProLParser::RuleSubspace);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(135);
    match(TAProLParser::T__10);
    setState(136);
    match(TAProLParser::T__3);
    setState(138);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAProLParser::ID) {
      setState(137);
      dynamic_cast<SubspaceContext *>(_localctx)->spacename = id();
    }
    setState(140);
    match(TAProLParser::T__4);
    setState(141);
    match(TAProLParser::T__9);
    setState(142);
    spacelist();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SpacelistContext ------------------------------------------------------------------

TAProLParser::SpacelistContext::SpacelistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::RangeContext* TAProLParser::SpacelistContext::range() {
  return getRuleContext<TAProLParser::RangeContext>(0);
}

TAProLParser::IdContext* TAProLParser::SpacelistContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
}

std::vector<TAProLParser::SpacelistContext *> TAProLParser::SpacelistContext::spacelist() {
  return getRuleContexts<TAProLParser::SpacelistContext>();
}

TAProLParser::SpacelistContext* TAProLParser::SpacelistContext::spacelist(size_t i) {
  return getRuleContext<TAProLParser::SpacelistContext>(i);
}


size_t TAProLParser::SpacelistContext::getRuleIndex() const {
  return TAProLParser::RuleSpacelist;
}

void TAProLParser::SpacelistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSpacelist(this);
}

void TAProLParser::SpacelistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSpacelist(this);
}

TAProLParser::SpacelistContext* TAProLParser::spacelist() {
  SpacelistContext *_localctx = _tracker.createInstance<SpacelistContext>(_ctx, getState());
  enterRule(_localctx, 20, TAProLParser::RuleSpacelist);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(144);
    dynamic_cast<SpacelistContext *>(_localctx)->spacename = id();
    setState(145);
    match(TAProLParser::T__11);
    setState(146);
    range();
    setState(149); 
    _errHandler->sync(this);
    alt = 1 + 1;
    do {
      switch (alt) {
        case 1 + 1: {
              setState(147);
              match(TAProLParser::T__12);
              setState(148);
              spacelist();
              break;
            }

      default:
        throw NoViableAltException(this);
      }
      setState(151); 
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 9, _ctx);
    } while (alt != 1 && alt != atn::ATN::INVALID_ALT_NUMBER);
   
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

std::vector<TAProLParser::IdxContext *> TAProLParser::IndexContext::idx() {
  return getRuleContexts<TAProLParser::IdxContext>();
}

TAProLParser::IdxContext* TAProLParser::IndexContext::idx(size_t i) {
  return getRuleContext<TAProLParser::IdxContext>(i);
}

TAProLParser::IdContext* TAProLParser::IndexContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
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
  enterRule(_localctx, 22, TAProLParser::RuleIndex);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(153);
    match(TAProLParser::T__13);
    setState(154);
    match(TAProLParser::T__3);
    setState(155);
    dynamic_cast<IndexContext *>(_localctx)->subspacename = id();
    setState(156);
    match(TAProLParser::T__4);
    setState(157);
    match(TAProLParser::T__9);
    setState(158);
    idx();
    setState(161); 
    _errHandler->sync(this);
    alt = 1 + 1;
    do {
      switch (alt) {
        case 1 + 1: {
              setState(159);
              match(TAProLParser::T__12);
              setState(160);
              idx();
              break;
            }

      default:
        throw NoViableAltException(this);
      }
      setState(163); 
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 10, _ctx);
    } while (alt != 1 && alt != atn::ATN::INVALID_ALT_NUMBER);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IdxContext ------------------------------------------------------------------

TAProLParser::IdxContext::IdxContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::IdContext* TAProLParser::IdxContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
}


size_t TAProLParser::IdxContext::getRuleIndex() const {
  return TAProLParser::RuleIdx;
}

void TAProLParser::IdxContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIdx(this);
}

void TAProLParser::IdxContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIdx(this);
}

TAProLParser::IdxContext* TAProLParser::idx() {
  IdxContext *_localctx = _tracker.createInstance<IdxContext>(_ctx, getState());
  enterRule(_localctx, 24, TAProLParser::RuleIdx);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(165);
    id();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AssignmentContext ------------------------------------------------------------------

TAProLParser::AssignmentContext::AssignmentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::TensorContext* TAProLParser::AssignmentContext::tensor() {
  return getRuleContext<TAProLParser::TensorContext>(0);
}

TAProLParser::StringContext* TAProLParser::AssignmentContext::string() {
  return getRuleContext<TAProLParser::StringContext>(0);
}

TAProLParser::ComplexContext* TAProLParser::AssignmentContext::complex() {
  return getRuleContext<TAProLParser::ComplexContext>(0);
}


size_t TAProLParser::AssignmentContext::getRuleIndex() const {
  return TAProLParser::RuleAssignment;
}

void TAProLParser::AssignmentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssignment(this);
}

void TAProLParser::AssignmentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssignment(this);
}

TAProLParser::AssignmentContext* TAProLParser::assignment() {
  AssignmentContext *_localctx = _tracker.createInstance<AssignmentContext>(_ctx, getState());
  enterRule(_localctx, 26, TAProLParser::RuleAssignment);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(182);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(167);
      tensor();
      setState(168);
      match(TAProLParser::T__11);
      setState(169);
      match(TAProLParser::T__14);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(171);
      tensor();
      setState(172);
      match(TAProLParser::T__11);
      setState(173);
      match(TAProLParser::T__15);
      setState(174);
      match(TAProLParser::T__3);
      setState(175);
      string();
      setState(176);
      match(TAProLParser::T__4);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(178);
      tensor();
      setState(179);
      match(TAProLParser::T__11);
      setState(180);
      complex();
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

TAProLParser::TensorContext* TAProLParser::LoadContext::tensor() {
  return getRuleContext<TAProLParser::TensorContext>(0);
}

TAProLParser::StringContext* TAProLParser::LoadContext::string() {
  return getRuleContext<TAProLParser::StringContext>(0);
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
  enterRule(_localctx, 28, TAProLParser::RuleLoad);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(184);
    match(TAProLParser::T__16);
    setState(185);
    tensor();
    setState(186);
    match(TAProLParser::T__9);
    setState(187);
    match(TAProLParser::T__17);
    setState(188);
    match(TAProLParser::T__3);
    setState(189);
    string();
    setState(190);
    match(TAProLParser::T__4);
   
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

TAProLParser::TensorContext* TAProLParser::SaveContext::tensor() {
  return getRuleContext<TAProLParser::TensorContext>(0);
}

TAProLParser::StringContext* TAProLParser::SaveContext::string() {
  return getRuleContext<TAProLParser::StringContext>(0);
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
  enterRule(_localctx, 30, TAProLParser::RuleSave);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(192);
    match(TAProLParser::T__18);
    setState(193);
    tensor();
    setState(194);
    match(TAProLParser::T__9);
    setState(195);
    match(TAProLParser::T__17);
    setState(196);
    match(TAProLParser::T__3);
    setState(197);
    string();
    setState(198);
    match(TAProLParser::T__4);
   
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

std::vector<TAProLParser::TensornameContext *> TAProLParser::DestroyContext::tensorname() {
  return getRuleContexts<TAProLParser::TensornameContext>();
}

TAProLParser::TensornameContext* TAProLParser::DestroyContext::tensorname(size_t i) {
  return getRuleContext<TAProLParser::TensornameContext>(i);
}

std::vector<TAProLParser::TensorContext *> TAProLParser::DestroyContext::tensor() {
  return getRuleContexts<TAProLParser::TensorContext>();
}

TAProLParser::TensorContext* TAProLParser::DestroyContext::tensor(size_t i) {
  return getRuleContext<TAProLParser::TensorContext>(i);
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
  enterRule(_localctx, 32, TAProLParser::RuleDestroy);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(216);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 15, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(200);
      match(TAProLParser::T__19);
      setState(201);
      tensorname();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(202);
      match(TAProLParser::T__19);
      setState(203);
      tensor();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(204);
      match(TAProLParser::T__20);
      setState(207);
      _errHandler->sync(this);
      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx)) {
      case 1: {
        setState(205);
        tensorname();
        break;
      }

      case 2: {
        setState(206);
        tensor();
        break;
      }

      }
      setState(214);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == TAProLParser::T__12) {
        setState(209);
        match(TAProLParser::T__12);
        setState(212);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 13, _ctx)) {
        case 1: {
          setState(210);
          tensorname();
          break;
        }

        case 2: {
          setState(211);
          tensor();
          break;
        }

        }
      }
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
  enterRule(_localctx, 34, TAProLParser::RuleCopy);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(218);
    tensor();
    setState(219);
    match(TAProLParser::T__11);
    setState(220);
    tensor();
   
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

TAProLParser::RealContext* TAProLParser::ScaleContext::real() {
  return getRuleContext<TAProLParser::RealContext>(0);
}

TAProLParser::ComplexContext* TAProLParser::ScaleContext::complex() {
  return getRuleContext<TAProLParser::ComplexContext>(0);
}

TAProLParser::IdContext* TAProLParser::ScaleContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
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
  enterRule(_localctx, 36, TAProLParser::RuleScale);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(222);
    tensor();
    setState(223);
    match(TAProLParser::T__21);
    setState(227);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TAProLParser::REAL: {
        setState(224);
        real();
        break;
      }

      case TAProLParser::T__28: {
        setState(225);
        complex();
        break;
      }

      case TAProLParser::ID: {
        setState(226);
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

//----------------- UnaryopContext ------------------------------------------------------------------

TAProLParser::UnaryopContext::UnaryopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAProLParser::TensorContext *> TAProLParser::UnaryopContext::tensor() {
  return getRuleContexts<TAProLParser::TensorContext>();
}

TAProLParser::TensorContext* TAProLParser::UnaryopContext::tensor(size_t i) {
  return getRuleContext<TAProLParser::TensorContext>(i);
}

TAProLParser::ConjtensorContext* TAProLParser::UnaryopContext::conjtensor() {
  return getRuleContext<TAProLParser::ConjtensorContext>(0);
}

TAProLParser::RealContext* TAProLParser::UnaryopContext::real() {
  return getRuleContext<TAProLParser::RealContext>(0);
}

TAProLParser::ComplexContext* TAProLParser::UnaryopContext::complex() {
  return getRuleContext<TAProLParser::ComplexContext>(0);
}

TAProLParser::IdContext* TAProLParser::UnaryopContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
}


size_t TAProLParser::UnaryopContext::getRuleIndex() const {
  return TAProLParser::RuleUnaryop;
}

void TAProLParser::UnaryopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnaryop(this);
}

void TAProLParser::UnaryopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnaryop(this);
}

TAProLParser::UnaryopContext* TAProLParser::unaryop() {
  UnaryopContext *_localctx = _tracker.createInstance<UnaryopContext>(_ctx, getState());
  enterRule(_localctx, 38, TAProLParser::RuleUnaryop);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(229);
    tensor();
    setState(230);
    match(TAProLParser::T__22);
    setState(233);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 17, _ctx)) {
    case 1: {
      setState(231);
      tensor();
      break;
    }

    case 2: {
      setState(232);
      conjtensor();
      break;
    }

    }
    setState(241);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAProLParser::T__23) {
      setState(235);
      match(TAProLParser::T__23);
      setState(239);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case TAProLParser::REAL: {
          setState(236);
          real();
          break;
        }

        case TAProLParser::T__28: {
          setState(237);
          complex();
          break;
        }

        case TAProLParser::ID: {
          setState(238);
          id();
          break;
        }

      default:
        throw NoViableAltException(this);
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

//----------------- BinaryopContext ------------------------------------------------------------------

TAProLParser::BinaryopContext::BinaryopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAProLParser::TensorContext *> TAProLParser::BinaryopContext::tensor() {
  return getRuleContexts<TAProLParser::TensorContext>();
}

TAProLParser::TensorContext* TAProLParser::BinaryopContext::tensor(size_t i) {
  return getRuleContext<TAProLParser::TensorContext>(i);
}

std::vector<TAProLParser::ConjtensorContext *> TAProLParser::BinaryopContext::conjtensor() {
  return getRuleContexts<TAProLParser::ConjtensorContext>();
}

TAProLParser::ConjtensorContext* TAProLParser::BinaryopContext::conjtensor(size_t i) {
  return getRuleContext<TAProLParser::ConjtensorContext>(i);
}

TAProLParser::RealContext* TAProLParser::BinaryopContext::real() {
  return getRuleContext<TAProLParser::RealContext>(0);
}

TAProLParser::ComplexContext* TAProLParser::BinaryopContext::complex() {
  return getRuleContext<TAProLParser::ComplexContext>(0);
}

TAProLParser::IdContext* TAProLParser::BinaryopContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
}


size_t TAProLParser::BinaryopContext::getRuleIndex() const {
  return TAProLParser::RuleBinaryop;
}

void TAProLParser::BinaryopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBinaryop(this);
}

void TAProLParser::BinaryopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBinaryop(this);
}

TAProLParser::BinaryopContext* TAProLParser::binaryop() {
  BinaryopContext *_localctx = _tracker.createInstance<BinaryopContext>(_ctx, getState());
  enterRule(_localctx, 40, TAProLParser::RuleBinaryop);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(243);
    tensor();
    setState(244);
    match(TAProLParser::T__22);
    setState(247);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 20, _ctx)) {
    case 1: {
      setState(245);
      tensor();
      break;
    }

    case 2: {
      setState(246);
      conjtensor();
      break;
    }

    }
    setState(249);
    match(TAProLParser::T__23);
    setState(252);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 21, _ctx)) {
    case 1: {
      setState(250);
      tensor();
      break;
    }

    case 2: {
      setState(251);
      conjtensor();
      break;
    }

    }
    setState(260);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAProLParser::T__23) {
      setState(254);
      match(TAProLParser::T__23);
      setState(258);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case TAProLParser::REAL: {
          setState(255);
          real();
          break;
        }

        case TAProLParser::T__28: {
          setState(256);
          complex();
          break;
        }

        case TAProLParser::ID: {
          setState(257);
          id();
          break;
        }

      default:
        throw NoViableAltException(this);
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

TAProLParser::RealContext* TAProLParser::CompositeproductContext::real() {
  return getRuleContext<TAProLParser::RealContext>(0);
}

TAProLParser::ComplexContext* TAProLParser::CompositeproductContext::complex() {
  return getRuleContext<TAProLParser::ComplexContext>(0);
}

TAProLParser::IdContext* TAProLParser::CompositeproductContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
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
  enterRule(_localctx, 42, TAProLParser::RuleCompositeproduct);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(262);
    tensor();
    setState(263);
    match(TAProLParser::T__22);
    setState(266);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx)) {
    case 1: {
      setState(264);
      tensor();
      break;
    }

    case 2: {
      setState(265);
      conjtensor();
      break;
    }

    }
    setState(268);
    match(TAProLParser::T__23);
    setState(271);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 25, _ctx)) {
    case 1: {
      setState(269);
      tensor();
      break;
    }

    case 2: {
      setState(270);
      conjtensor();
      break;
    }

    }
    setState(278); 
    _errHandler->sync(this);
    alt = 1;
    do {
      switch (alt) {
        case 1: {
              setState(273);
              match(TAProLParser::T__23);
              setState(276);
              _errHandler->sync(this);
              switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 26, _ctx)) {
              case 1: {
                setState(274);
                tensor();
                break;
              }

              case 2: {
                setState(275);
                conjtensor();
                break;
              }

              }
              break;
            }

      default:
        throw NoViableAltException(this);
      }
      setState(280); 
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 27, _ctx);
    } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
    setState(288);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAProLParser::T__23) {
      setState(282);
      match(TAProLParser::T__23);
      setState(286);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case TAProLParser::REAL: {
          setState(283);
          real();
          break;
        }

        case TAProLParser::T__28: {
          setState(284);
          complex();
          break;
        }

        case TAProLParser::ID: {
          setState(285);
          id();
          break;
        }

      default:
        throw NoViableAltException(this);
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
  enterRule(_localctx, 44, TAProLParser::RuleTensornetwork);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(290);
    tensor();
    setState(291);
    match(TAProLParser::T__24);
    setState(294);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 30, _ctx)) {
    case 1: {
      setState(292);
      tensor();
      break;
    }

    case 2: {
      setState(293);
      conjtensor();
      break;
    }

    }
    setState(301); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(296);
      match(TAProLParser::T__23);
      setState(299);
      _errHandler->sync(this);
      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 31, _ctx)) {
      case 1: {
        setState(297);
        tensor();
        break;
      }

      case 2: {
        setState(298);
        conjtensor();
        break;
      }

      }
      setState(303); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == TAProLParser::T__23);
   
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
  enterRule(_localctx, 46, TAProLParser::RuleTensorname);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(305);
    id();
   
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
  enterRule(_localctx, 48, TAProLParser::RuleTensor);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(315);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 34, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(307);
      tensorname();
      setState(308);
      match(TAProLParser::T__3);
      setState(310);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == TAProLParser::ID

      || _la == TAProLParser::INT) {
        setState(309);
        indexlist();
      }
      setState(312);
      match(TAProLParser::T__4);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(314);
      tensorname();
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
  enterRule(_localctx, 50, TAProLParser::RuleConjtensor);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(317);
    tensorname();
    setState(318);
    match(TAProLParser::T__25);
    setState(319);
    match(TAProLParser::T__3);
    setState(321);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAProLParser::ID

    || _la == TAProLParser::INT) {
      setState(320);
      indexlist();
    }
    setState(323);
    match(TAProLParser::T__4);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ActualindexContext ------------------------------------------------------------------

TAProLParser::ActualindexContext::ActualindexContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAProLParser::IdContext* TAProLParser::ActualindexContext::id() {
  return getRuleContext<TAProLParser::IdContext>(0);
}

tree::TerminalNode* TAProLParser::ActualindexContext::INT() {
  return getToken(TAProLParser::INT, 0);
}


size_t TAProLParser::ActualindexContext::getRuleIndex() const {
  return TAProLParser::RuleActualindex;
}

void TAProLParser::ActualindexContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterActualindex(this);
}

void TAProLParser::ActualindexContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TAProLListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitActualindex(this);
}

TAProLParser::ActualindexContext* TAProLParser::actualindex() {
  ActualindexContext *_localctx = _tracker.createInstance<ActualindexContext>(_ctx, getState());
  enterRule(_localctx, 52, TAProLParser::RuleActualindex);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(327);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TAProLParser::ID: {
        enterOuterAlt(_localctx, 1);
        setState(325);
        id();
        break;
      }

      case TAProLParser::INT: {
        enterOuterAlt(_localctx, 2);
        setState(326);
        match(TAProLParser::INT);
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

//----------------- IndexlistContext ------------------------------------------------------------------

TAProLParser::IndexlistContext::IndexlistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAProLParser::ActualindexContext *> TAProLParser::IndexlistContext::actualindex() {
  return getRuleContexts<TAProLParser::ActualindexContext>();
}

TAProLParser::ActualindexContext* TAProLParser::IndexlistContext::actualindex(size_t i) {
  return getRuleContext<TAProLParser::ActualindexContext>(i);
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
  enterRule(_localctx, 54, TAProLParser::RuleIndexlist);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(329);
    actualindex();
    setState(332); 
    _errHandler->sync(this);
    alt = 1 + 1;
    do {
      switch (alt) {
        case 1 + 1: {
              setState(330);
              match(TAProLParser::T__12);
              setState(331);
              actualindex();
              break;
            }

      default:
        throw NoViableAltException(this);
      }
      setState(334); 
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 37, _ctx);
    } while (alt != 1 && alt != atn::ATN::INVALID_ALT_NUMBER);
   
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
  enterRule(_localctx, 56, TAProLParser::RuleComment);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(336);
    match(TAProLParser::COMMENT);
   
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

std::vector<tree::TerminalNode *> TAProLParser::RangeContext::INT() {
  return getTokens(TAProLParser::INT);
}

tree::TerminalNode* TAProLParser::RangeContext::INT(size_t i) {
  return getToken(TAProLParser::INT, i);
}

std::vector<TAProLParser::IdContext *> TAProLParser::RangeContext::id() {
  return getRuleContexts<TAProLParser::IdContext>();
}

TAProLParser::IdContext* TAProLParser::RangeContext::id(size_t i) {
  return getRuleContext<TAProLParser::IdContext>(i);
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
  enterRule(_localctx, 58, TAProLParser::RuleRange);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(338);
    match(TAProLParser::T__26);
    setState(341);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TAProLParser::INT: {
        setState(339);
        match(TAProLParser::INT);
        break;
      }

      case TAProLParser::ID: {
        setState(340);
        id();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    setState(343);
    match(TAProLParser::T__9);
    setState(346);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TAProLParser::INT: {
        setState(344);
        match(TAProLParser::INT);
        break;
      }

      case TAProLParser::ID: {
        setState(345);
        id();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    setState(348);
    match(TAProLParser::T__27);
   
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

TAProLParser::GroupnameContext* TAProLParser::GroupnamelistContext::groupname() {
  return getRuleContext<TAProLParser::GroupnameContext>(0);
}

TAProLParser::GroupnamelistContext* TAProLParser::GroupnamelistContext::groupnamelist() {
  return getRuleContext<TAProLParser::GroupnamelistContext>(0);
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
  enterRule(_localctx, 60, TAProLParser::RuleGroupnamelist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(350);
    groupname();
    setState(353);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAProLParser::T__12) {
      setState(351);
      match(TAProLParser::T__12);
      setState(352);
      groupnamelist();
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
  enterRule(_localctx, 62, TAProLParser::RuleGroupname);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(355);
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
  enterRule(_localctx, 64, TAProLParser::RuleId);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(357);
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
  enterRule(_localctx, 66, TAProLParser::RuleComplex);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(359);
    match(TAProLParser::T__28);
    setState(360);
    real();
    setState(361);
    match(TAProLParser::T__12);
    setState(362);
    real();
    setState(363);
    match(TAProLParser::T__29);
   
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
  enterRule(_localctx, 68, TAProLParser::RuleReal);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(365);
    match(TAProLParser::REAL);
   
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
  enterRule(_localctx, 70, TAProLParser::RuleString);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(367);
    match(TAProLParser::STRING);
   
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
  "taprolsrc", "entry", "scope", "code", "line", "statement", "simpleop", 
  "compositeop", "space", "subspace", "spacelist", "index", "idx", "assignment", 
  "load", "save", "destroy", "copy", "scale", "unaryop", "binaryop", "compositeproduct", 
  "tensornetwork", "tensorname", "tensor", "conjtensor", "actualindex", 
  "indexlist", "comment", "range", "groupnamelist", "groupname", "id", "complex", 
  "real", "string"
};

std::vector<std::string> TAProLParser::_literalNames = {
  "", "'entry:'", "'scope'", "'group'", "'('", "')'", "'end'", "'space'", 
  "'real'", "'complex'", "':'", "'subspace'", "'='", "','", "'index'", "'?'", 
  "'method'", "'load'", "'tag'", "'save'", "'~'", "'destroy'", "'*='", "'+='", 
  "'*'", "'=>'", "'+'", "'['", "']'", "'{'", "'}'"
};

std::vector<std::string> TAProLParser::_symbolicNames = {
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "COMMENT", "ID", "REAL", 
  "INT", "STRING", "WS", "EOL"
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
    0x3, 0x27, 0x174, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
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
    0x25, 0x3, 0x2, 0x3, 0x2, 0x6, 0x2, 0x4d, 0xa, 0x2, 0xd, 0x2, 0xe, 0x2, 
    0x4e, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 
    0x4, 0x3, 0x4, 0x5, 0x4, 0x59, 0xa, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 
    0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x5, 0x6, 0x5, 0x62, 0xa, 0x5, 0xd, 
    0x5, 0xe, 0x5, 0x63, 0x3, 0x6, 0x6, 0x6, 0x67, 0xa, 0x6, 0xd, 0x6, 0xe, 
    0x6, 0x68, 0x3, 0x6, 0x5, 0x6, 0x6c, 0xa, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 
    0x7, 0x3, 0x7, 0x3, 0x7, 0x5, 0x7, 0x73, 0xa, 0x7, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 
    0x7d, 0xa, 0x8, 0x3, 0x9, 0x3, 0x9, 0x5, 0x9, 0x81, 0xa, 0x9, 0x3, 0xa, 
    0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 0x8d, 0xa, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 
    0xb, 0x3, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x6, 
    0xc, 0x98, 0xa, 0xc, 0xd, 0xc, 0xe, 0xc, 0x99, 0x3, 0xd, 0x3, 0xd, 0x3, 
    0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x6, 0xd, 0xa4, 
    0xa, 0xd, 0xd, 0xd, 0xe, 0xd, 0xa5, 0x3, 0xe, 0x3, 0xe, 0x3, 0xf, 0x3, 
    0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 
    0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x5, 
    0xf, 0xb9, 0xa, 0xf, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 
    0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 
    0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x12, 0x3, 
    0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x5, 0x12, 
    0xd2, 0xa, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x5, 0x12, 0xd7, 0xa, 
    0x12, 0x5, 0x12, 0xd9, 0xa, 0x12, 0x5, 0x12, 0xdb, 0xa, 0x12, 0x3, 0x13, 
    0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x5, 0x14, 0xe6, 0xa, 0x14, 0x3, 0x15, 0x3, 0x15, 0x3, 
    0x15, 0x3, 0x15, 0x5, 0x15, 0xec, 0xa, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 
    0x15, 0x3, 0x15, 0x5, 0x15, 0xf2, 0xa, 0x15, 0x5, 0x15, 0xf4, 0xa, 0x15, 
    0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x5, 0x16, 0xfa, 0xa, 0x16, 
    0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x5, 0x16, 0xff, 0xa, 0x16, 0x3, 0x16, 
    0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x5, 0x16, 0x105, 0xa, 0x16, 0x5, 0x16, 
    0x107, 0xa, 0x16, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x5, 0x17, 
    0x10d, 0xa, 0x17, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x5, 0x17, 0x112, 
    0xa, 0x17, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x5, 0x17, 0x117, 0xa, 0x17, 
    0x6, 0x17, 0x119, 0xa, 0x17, 0xd, 0x17, 0xe, 0x17, 0x11a, 0x3, 0x17, 
    0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x5, 0x17, 0x121, 0xa, 0x17, 0x5, 0x17, 
    0x123, 0xa, 0x17, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x5, 0x18, 
    0x129, 0xa, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x5, 0x18, 0x12e, 
    0xa, 0x18, 0x6, 0x18, 0x130, 0xa, 0x18, 0xd, 0x18, 0xe, 0x18, 0x131, 
    0x3, 0x19, 0x3, 0x19, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x5, 0x1a, 0x139, 
    0xa, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x5, 0x1a, 0x13e, 0xa, 0x1a, 
    0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x5, 0x1b, 0x144, 0xa, 0x1b, 
    0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1c, 0x3, 0x1c, 0x5, 0x1c, 0x14a, 0xa, 0x1c, 
    0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x6, 0x1d, 0x14f, 0xa, 0x1d, 0xd, 0x1d, 
    0xe, 0x1d, 0x150, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 
    0x5, 0x1f, 0x158, 0xa, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x5, 0x1f, 
    0x15d, 0xa, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 
    0x5, 0x20, 0x164, 0xa, 0x20, 0x3, 0x21, 0x3, 0x21, 0x3, 0x22, 0x3, 0x22, 
    0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 
    0x24, 0x3, 0x24, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x5, 0x99, 0xa5, 0x150, 
    0x2, 0x26, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 
    0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 
    0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e, 0x40, 0x42, 0x44, 0x46, 
    0x48, 0x2, 0x3, 0x3, 0x2, 0xa, 0xb, 0x2, 0x187, 0x2, 0x4a, 0x3, 0x2, 
    0x2, 0x2, 0x4, 0x50, 0x3, 0x2, 0x2, 0x2, 0x6, 0x53, 0x3, 0x2, 0x2, 0x2, 
    0x8, 0x61, 0x3, 0x2, 0x2, 0x2, 0xa, 0x6b, 0x3, 0x2, 0x2, 0x2, 0xc, 0x72, 
    0x3, 0x2, 0x2, 0x2, 0xe, 0x7c, 0x3, 0x2, 0x2, 0x2, 0x10, 0x80, 0x3, 
    0x2, 0x2, 0x2, 0x12, 0x82, 0x3, 0x2, 0x2, 0x2, 0x14, 0x89, 0x3, 0x2, 
    0x2, 0x2, 0x16, 0x92, 0x3, 0x2, 0x2, 0x2, 0x18, 0x9b, 0x3, 0x2, 0x2, 
    0x2, 0x1a, 0xa7, 0x3, 0x2, 0x2, 0x2, 0x1c, 0xb8, 0x3, 0x2, 0x2, 0x2, 
    0x1e, 0xba, 0x3, 0x2, 0x2, 0x2, 0x20, 0xc2, 0x3, 0x2, 0x2, 0x2, 0x22, 
    0xda, 0x3, 0x2, 0x2, 0x2, 0x24, 0xdc, 0x3, 0x2, 0x2, 0x2, 0x26, 0xe0, 
    0x3, 0x2, 0x2, 0x2, 0x28, 0xe7, 0x3, 0x2, 0x2, 0x2, 0x2a, 0xf5, 0x3, 
    0x2, 0x2, 0x2, 0x2c, 0x108, 0x3, 0x2, 0x2, 0x2, 0x2e, 0x124, 0x3, 0x2, 
    0x2, 0x2, 0x30, 0x133, 0x3, 0x2, 0x2, 0x2, 0x32, 0x13d, 0x3, 0x2, 0x2, 
    0x2, 0x34, 0x13f, 0x3, 0x2, 0x2, 0x2, 0x36, 0x149, 0x3, 0x2, 0x2, 0x2, 
    0x38, 0x14b, 0x3, 0x2, 0x2, 0x2, 0x3a, 0x152, 0x3, 0x2, 0x2, 0x2, 0x3c, 
    0x154, 0x3, 0x2, 0x2, 0x2, 0x3e, 0x160, 0x3, 0x2, 0x2, 0x2, 0x40, 0x165, 
    0x3, 0x2, 0x2, 0x2, 0x42, 0x167, 0x3, 0x2, 0x2, 0x2, 0x44, 0x169, 0x3, 
    0x2, 0x2, 0x2, 0x46, 0x16f, 0x3, 0x2, 0x2, 0x2, 0x48, 0x171, 0x3, 0x2, 
    0x2, 0x2, 0x4a, 0x4c, 0x5, 0x4, 0x3, 0x2, 0x4b, 0x4d, 0x5, 0x6, 0x4, 
    0x2, 0x4c, 0x4b, 0x3, 0x2, 0x2, 0x2, 0x4d, 0x4e, 0x3, 0x2, 0x2, 0x2, 
    0x4e, 0x4c, 0x3, 0x2, 0x2, 0x2, 0x4e, 0x4f, 0x3, 0x2, 0x2, 0x2, 0x4f, 
    0x3, 0x3, 0x2, 0x2, 0x2, 0x50, 0x51, 0x7, 0x3, 0x2, 0x2, 0x51, 0x52, 
    0x7, 0x22, 0x2, 0x2, 0x52, 0x5, 0x3, 0x2, 0x2, 0x2, 0x53, 0x54, 0x7, 
    0x4, 0x2, 0x2, 0x54, 0x55, 0x5, 0x42, 0x22, 0x2, 0x55, 0x56, 0x7, 0x5, 
    0x2, 0x2, 0x56, 0x58, 0x7, 0x6, 0x2, 0x2, 0x57, 0x59, 0x5, 0x3e, 0x20, 
    0x2, 0x58, 0x57, 0x3, 0x2, 0x2, 0x2, 0x58, 0x59, 0x3, 0x2, 0x2, 0x2, 
    0x59, 0x5a, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x5b, 0x7, 0x7, 0x2, 0x2, 0x5b, 
    0x5c, 0x5, 0x8, 0x5, 0x2, 0x5c, 0x5d, 0x7, 0x8, 0x2, 0x2, 0x5d, 0x5e, 
    0x7, 0x4, 0x2, 0x2, 0x5e, 0x5f, 0x5, 0x42, 0x22, 0x2, 0x5f, 0x7, 0x3, 
    0x2, 0x2, 0x2, 0x60, 0x62, 0x5, 0xa, 0x6, 0x2, 0x61, 0x60, 0x3, 0x2, 
    0x2, 0x2, 0x62, 0x63, 0x3, 0x2, 0x2, 0x2, 0x63, 0x61, 0x3, 0x2, 0x2, 
    0x2, 0x63, 0x64, 0x3, 0x2, 0x2, 0x2, 0x64, 0x9, 0x3, 0x2, 0x2, 0x2, 
    0x65, 0x67, 0x5, 0xc, 0x7, 0x2, 0x66, 0x65, 0x3, 0x2, 0x2, 0x2, 0x67, 
    0x68, 0x3, 0x2, 0x2, 0x2, 0x68, 0x66, 0x3, 0x2, 0x2, 0x2, 0x68, 0x69, 
    0x3, 0x2, 0x2, 0x2, 0x69, 0x6c, 0x3, 0x2, 0x2, 0x2, 0x6a, 0x6c, 0x5, 
    0x3a, 0x1e, 0x2, 0x6b, 0x66, 0x3, 0x2, 0x2, 0x2, 0x6b, 0x6a, 0x3, 0x2, 
    0x2, 0x2, 0x6c, 0xb, 0x3, 0x2, 0x2, 0x2, 0x6d, 0x73, 0x5, 0x12, 0xa, 
    0x2, 0x6e, 0x73, 0x5, 0x14, 0xb, 0x2, 0x6f, 0x73, 0x5, 0x18, 0xd, 0x2, 
    0x70, 0x73, 0x5, 0xe, 0x8, 0x2, 0x71, 0x73, 0x5, 0x10, 0x9, 0x2, 0x72, 
    0x6d, 0x3, 0x2, 0x2, 0x2, 0x72, 0x6e, 0x3, 0x2, 0x2, 0x2, 0x72, 0x6f, 
    0x3, 0x2, 0x2, 0x2, 0x72, 0x70, 0x3, 0x2, 0x2, 0x2, 0x72, 0x71, 0x3, 
    0x2, 0x2, 0x2, 0x73, 0xd, 0x3, 0x2, 0x2, 0x2, 0x74, 0x7d, 0x5, 0x1c, 
    0xf, 0x2, 0x75, 0x7d, 0x5, 0x1e, 0x10, 0x2, 0x76, 0x7d, 0x5, 0x20, 0x11, 
    0x2, 0x77, 0x7d, 0x5, 0x22, 0x12, 0x2, 0x78, 0x7d, 0x5, 0x24, 0x13, 
    0x2, 0x79, 0x7d, 0x5, 0x26, 0x14, 0x2, 0x7a, 0x7d, 0x5, 0x28, 0x15, 
    0x2, 0x7b, 0x7d, 0x5, 0x2a, 0x16, 0x2, 0x7c, 0x74, 0x3, 0x2, 0x2, 0x2, 
    0x7c, 0x75, 0x3, 0x2, 0x2, 0x2, 0x7c, 0x76, 0x3, 0x2, 0x2, 0x2, 0x7c, 
    0x77, 0x3, 0x2, 0x2, 0x2, 0x7c, 0x78, 0x3, 0x2, 0x2, 0x2, 0x7c, 0x79, 
    0x3, 0x2, 0x2, 0x2, 0x7c, 0x7a, 0x3, 0x2, 0x2, 0x2, 0x7c, 0x7b, 0x3, 
    0x2, 0x2, 0x2, 0x7d, 0xf, 0x3, 0x2, 0x2, 0x2, 0x7e, 0x81, 0x5, 0x2c, 
    0x17, 0x2, 0x7f, 0x81, 0x5, 0x2e, 0x18, 0x2, 0x80, 0x7e, 0x3, 0x2, 0x2, 
    0x2, 0x80, 0x7f, 0x3, 0x2, 0x2, 0x2, 0x81, 0x11, 0x3, 0x2, 0x2, 0x2, 
    0x82, 0x83, 0x7, 0x9, 0x2, 0x2, 0x83, 0x84, 0x7, 0x6, 0x2, 0x2, 0x84, 
    0x85, 0x9, 0x2, 0x2, 0x2, 0x85, 0x86, 0x7, 0x7, 0x2, 0x2, 0x86, 0x87, 
    0x7, 0xc, 0x2, 0x2, 0x87, 0x88, 0x5, 0x16, 0xc, 0x2, 0x88, 0x13, 0x3, 
    0x2, 0x2, 0x2, 0x89, 0x8a, 0x7, 0xd, 0x2, 0x2, 0x8a, 0x8c, 0x7, 0x6, 
    0x2, 0x2, 0x8b, 0x8d, 0x5, 0x42, 0x22, 0x2, 0x8c, 0x8b, 0x3, 0x2, 0x2, 
    0x2, 0x8c, 0x8d, 0x3, 0x2, 0x2, 0x2, 0x8d, 0x8e, 0x3, 0x2, 0x2, 0x2, 
    0x8e, 0x8f, 0x7, 0x7, 0x2, 0x2, 0x8f, 0x90, 0x7, 0xc, 0x2, 0x2, 0x90, 
    0x91, 0x5, 0x16, 0xc, 0x2, 0x91, 0x15, 0x3, 0x2, 0x2, 0x2, 0x92, 0x93, 
    0x5, 0x42, 0x22, 0x2, 0x93, 0x94, 0x7, 0xe, 0x2, 0x2, 0x94, 0x97, 0x5, 
    0x3c, 0x1f, 0x2, 0x95, 0x96, 0x7, 0xf, 0x2, 0x2, 0x96, 0x98, 0x5, 0x16, 
    0xc, 0x2, 0x97, 0x95, 0x3, 0x2, 0x2, 0x2, 0x98, 0x99, 0x3, 0x2, 0x2, 
    0x2, 0x99, 0x9a, 0x3, 0x2, 0x2, 0x2, 0x99, 0x97, 0x3, 0x2, 0x2, 0x2, 
    0x9a, 0x17, 0x3, 0x2, 0x2, 0x2, 0x9b, 0x9c, 0x7, 0x10, 0x2, 0x2, 0x9c, 
    0x9d, 0x7, 0x6, 0x2, 0x2, 0x9d, 0x9e, 0x5, 0x42, 0x22, 0x2, 0x9e, 0x9f, 
    0x7, 0x7, 0x2, 0x2, 0x9f, 0xa0, 0x7, 0xc, 0x2, 0x2, 0xa0, 0xa3, 0x5, 
    0x1a, 0xe, 0x2, 0xa1, 0xa2, 0x7, 0xf, 0x2, 0x2, 0xa2, 0xa4, 0x5, 0x1a, 
    0xe, 0x2, 0xa3, 0xa1, 0x3, 0x2, 0x2, 0x2, 0xa4, 0xa5, 0x3, 0x2, 0x2, 
    0x2, 0xa5, 0xa6, 0x3, 0x2, 0x2, 0x2, 0xa5, 0xa3, 0x3, 0x2, 0x2, 0x2, 
    0xa6, 0x19, 0x3, 0x2, 0x2, 0x2, 0xa7, 0xa8, 0x5, 0x42, 0x22, 0x2, 0xa8, 
    0x1b, 0x3, 0x2, 0x2, 0x2, 0xa9, 0xaa, 0x5, 0x32, 0x1a, 0x2, 0xaa, 0xab, 
    0x7, 0xe, 0x2, 0x2, 0xab, 0xac, 0x7, 0x11, 0x2, 0x2, 0xac, 0xb9, 0x3, 
    0x2, 0x2, 0x2, 0xad, 0xae, 0x5, 0x32, 0x1a, 0x2, 0xae, 0xaf, 0x7, 0xe, 
    0x2, 0x2, 0xaf, 0xb0, 0x7, 0x12, 0x2, 0x2, 0xb0, 0xb1, 0x7, 0x6, 0x2, 
    0x2, 0xb1, 0xb2, 0x5, 0x48, 0x25, 0x2, 0xb2, 0xb3, 0x7, 0x7, 0x2, 0x2, 
    0xb3, 0xb9, 0x3, 0x2, 0x2, 0x2, 0xb4, 0xb5, 0x5, 0x32, 0x1a, 0x2, 0xb5, 
    0xb6, 0x7, 0xe, 0x2, 0x2, 0xb6, 0xb7, 0x5, 0x44, 0x23, 0x2, 0xb7, 0xb9, 
    0x3, 0x2, 0x2, 0x2, 0xb8, 0xa9, 0x3, 0x2, 0x2, 0x2, 0xb8, 0xad, 0x3, 
    0x2, 0x2, 0x2, 0xb8, 0xb4, 0x3, 0x2, 0x2, 0x2, 0xb9, 0x1d, 0x3, 0x2, 
    0x2, 0x2, 0xba, 0xbb, 0x7, 0x13, 0x2, 0x2, 0xbb, 0xbc, 0x5, 0x32, 0x1a, 
    0x2, 0xbc, 0xbd, 0x7, 0xc, 0x2, 0x2, 0xbd, 0xbe, 0x7, 0x14, 0x2, 0x2, 
    0xbe, 0xbf, 0x7, 0x6, 0x2, 0x2, 0xbf, 0xc0, 0x5, 0x48, 0x25, 0x2, 0xc0, 
    0xc1, 0x7, 0x7, 0x2, 0x2, 0xc1, 0x1f, 0x3, 0x2, 0x2, 0x2, 0xc2, 0xc3, 
    0x7, 0x15, 0x2, 0x2, 0xc3, 0xc4, 0x5, 0x32, 0x1a, 0x2, 0xc4, 0xc5, 0x7, 
    0xc, 0x2, 0x2, 0xc5, 0xc6, 0x7, 0x14, 0x2, 0x2, 0xc6, 0xc7, 0x7, 0x6, 
    0x2, 0x2, 0xc7, 0xc8, 0x5, 0x48, 0x25, 0x2, 0xc8, 0xc9, 0x7, 0x7, 0x2, 
    0x2, 0xc9, 0x21, 0x3, 0x2, 0x2, 0x2, 0xca, 0xcb, 0x7, 0x16, 0x2, 0x2, 
    0xcb, 0xdb, 0x5, 0x30, 0x19, 0x2, 0xcc, 0xcd, 0x7, 0x16, 0x2, 0x2, 0xcd, 
    0xdb, 0x5, 0x32, 0x1a, 0x2, 0xce, 0xd1, 0x7, 0x17, 0x2, 0x2, 0xcf, 0xd2, 
    0x5, 0x30, 0x19, 0x2, 0xd0, 0xd2, 0x5, 0x32, 0x1a, 0x2, 0xd1, 0xcf, 
    0x3, 0x2, 0x2, 0x2, 0xd1, 0xd0, 0x3, 0x2, 0x2, 0x2, 0xd2, 0xd8, 0x3, 
    0x2, 0x2, 0x2, 0xd3, 0xd6, 0x7, 0xf, 0x2, 0x2, 0xd4, 0xd7, 0x5, 0x30, 
    0x19, 0x2, 0xd5, 0xd7, 0x5, 0x32, 0x1a, 0x2, 0xd6, 0xd4, 0x3, 0x2, 0x2, 
    0x2, 0xd6, 0xd5, 0x3, 0x2, 0x2, 0x2, 0xd7, 0xd9, 0x3, 0x2, 0x2, 0x2, 
    0xd8, 0xd3, 0x3, 0x2, 0x2, 0x2, 0xd8, 0xd9, 0x3, 0x2, 0x2, 0x2, 0xd9, 
    0xdb, 0x3, 0x2, 0x2, 0x2, 0xda, 0xca, 0x3, 0x2, 0x2, 0x2, 0xda, 0xcc, 
    0x3, 0x2, 0x2, 0x2, 0xda, 0xce, 0x3, 0x2, 0x2, 0x2, 0xdb, 0x23, 0x3, 
    0x2, 0x2, 0x2, 0xdc, 0xdd, 0x5, 0x32, 0x1a, 0x2, 0xdd, 0xde, 0x7, 0xe, 
    0x2, 0x2, 0xde, 0xdf, 0x5, 0x32, 0x1a, 0x2, 0xdf, 0x25, 0x3, 0x2, 0x2, 
    0x2, 0xe0, 0xe1, 0x5, 0x32, 0x1a, 0x2, 0xe1, 0xe5, 0x7, 0x18, 0x2, 0x2, 
    0xe2, 0xe6, 0x5, 0x46, 0x24, 0x2, 0xe3, 0xe6, 0x5, 0x44, 0x23, 0x2, 
    0xe4, 0xe6, 0x5, 0x42, 0x22, 0x2, 0xe5, 0xe2, 0x3, 0x2, 0x2, 0x2, 0xe5, 
    0xe3, 0x3, 0x2, 0x2, 0x2, 0xe5, 0xe4, 0x3, 0x2, 0x2, 0x2, 0xe6, 0x27, 
    0x3, 0x2, 0x2, 0x2, 0xe7, 0xe8, 0x5, 0x32, 0x1a, 0x2, 0xe8, 0xeb, 0x7, 
    0x19, 0x2, 0x2, 0xe9, 0xec, 0x5, 0x32, 0x1a, 0x2, 0xea, 0xec, 0x5, 0x34, 
    0x1b, 0x2, 0xeb, 0xe9, 0x3, 0x2, 0x2, 0x2, 0xeb, 0xea, 0x3, 0x2, 0x2, 
    0x2, 0xec, 0xf3, 0x3, 0x2, 0x2, 0x2, 0xed, 0xf1, 0x7, 0x1a, 0x2, 0x2, 
    0xee, 0xf2, 0x5, 0x46, 0x24, 0x2, 0xef, 0xf2, 0x5, 0x44, 0x23, 0x2, 
    0xf0, 0xf2, 0x5, 0x42, 0x22, 0x2, 0xf1, 0xee, 0x3, 0x2, 0x2, 0x2, 0xf1, 
    0xef, 0x3, 0x2, 0x2, 0x2, 0xf1, 0xf0, 0x3, 0x2, 0x2, 0x2, 0xf2, 0xf4, 
    0x3, 0x2, 0x2, 0x2, 0xf3, 0xed, 0x3, 0x2, 0x2, 0x2, 0xf3, 0xf4, 0x3, 
    0x2, 0x2, 0x2, 0xf4, 0x29, 0x3, 0x2, 0x2, 0x2, 0xf5, 0xf6, 0x5, 0x32, 
    0x1a, 0x2, 0xf6, 0xf9, 0x7, 0x19, 0x2, 0x2, 0xf7, 0xfa, 0x5, 0x32, 0x1a, 
    0x2, 0xf8, 0xfa, 0x5, 0x34, 0x1b, 0x2, 0xf9, 0xf7, 0x3, 0x2, 0x2, 0x2, 
    0xf9, 0xf8, 0x3, 0x2, 0x2, 0x2, 0xfa, 0xfb, 0x3, 0x2, 0x2, 0x2, 0xfb, 
    0xfe, 0x7, 0x1a, 0x2, 0x2, 0xfc, 0xff, 0x5, 0x32, 0x1a, 0x2, 0xfd, 0xff, 
    0x5, 0x34, 0x1b, 0x2, 0xfe, 0xfc, 0x3, 0x2, 0x2, 0x2, 0xfe, 0xfd, 0x3, 
    0x2, 0x2, 0x2, 0xff, 0x106, 0x3, 0x2, 0x2, 0x2, 0x100, 0x104, 0x7, 0x1a, 
    0x2, 0x2, 0x101, 0x105, 0x5, 0x46, 0x24, 0x2, 0x102, 0x105, 0x5, 0x44, 
    0x23, 0x2, 0x103, 0x105, 0x5, 0x42, 0x22, 0x2, 0x104, 0x101, 0x3, 0x2, 
    0x2, 0x2, 0x104, 0x102, 0x3, 0x2, 0x2, 0x2, 0x104, 0x103, 0x3, 0x2, 
    0x2, 0x2, 0x105, 0x107, 0x3, 0x2, 0x2, 0x2, 0x106, 0x100, 0x3, 0x2, 
    0x2, 0x2, 0x106, 0x107, 0x3, 0x2, 0x2, 0x2, 0x107, 0x2b, 0x3, 0x2, 0x2, 
    0x2, 0x108, 0x109, 0x5, 0x32, 0x1a, 0x2, 0x109, 0x10c, 0x7, 0x19, 0x2, 
    0x2, 0x10a, 0x10d, 0x5, 0x32, 0x1a, 0x2, 0x10b, 0x10d, 0x5, 0x34, 0x1b, 
    0x2, 0x10c, 0x10a, 0x3, 0x2, 0x2, 0x2, 0x10c, 0x10b, 0x3, 0x2, 0x2, 
    0x2, 0x10d, 0x10e, 0x3, 0x2, 0x2, 0x2, 0x10e, 0x111, 0x7, 0x1a, 0x2, 
    0x2, 0x10f, 0x112, 0x5, 0x32, 0x1a, 0x2, 0x110, 0x112, 0x5, 0x34, 0x1b, 
    0x2, 0x111, 0x10f, 0x3, 0x2, 0x2, 0x2, 0x111, 0x110, 0x3, 0x2, 0x2, 
    0x2, 0x112, 0x118, 0x3, 0x2, 0x2, 0x2, 0x113, 0x116, 0x7, 0x1a, 0x2, 
    0x2, 0x114, 0x117, 0x5, 0x32, 0x1a, 0x2, 0x115, 0x117, 0x5, 0x34, 0x1b, 
    0x2, 0x116, 0x114, 0x3, 0x2, 0x2, 0x2, 0x116, 0x115, 0x3, 0x2, 0x2, 
    0x2, 0x117, 0x119, 0x3, 0x2, 0x2, 0x2, 0x118, 0x113, 0x3, 0x2, 0x2, 
    0x2, 0x119, 0x11a, 0x3, 0x2, 0x2, 0x2, 0x11a, 0x118, 0x3, 0x2, 0x2, 
    0x2, 0x11a, 0x11b, 0x3, 0x2, 0x2, 0x2, 0x11b, 0x122, 0x3, 0x2, 0x2, 
    0x2, 0x11c, 0x120, 0x7, 0x1a, 0x2, 0x2, 0x11d, 0x121, 0x5, 0x46, 0x24, 
    0x2, 0x11e, 0x121, 0x5, 0x44, 0x23, 0x2, 0x11f, 0x121, 0x5, 0x42, 0x22, 
    0x2, 0x120, 0x11d, 0x3, 0x2, 0x2, 0x2, 0x120, 0x11e, 0x3, 0x2, 0x2, 
    0x2, 0x120, 0x11f, 0x3, 0x2, 0x2, 0x2, 0x121, 0x123, 0x3, 0x2, 0x2, 
    0x2, 0x122, 0x11c, 0x3, 0x2, 0x2, 0x2, 0x122, 0x123, 0x3, 0x2, 0x2, 
    0x2, 0x123, 0x2d, 0x3, 0x2, 0x2, 0x2, 0x124, 0x125, 0x5, 0x32, 0x1a, 
    0x2, 0x125, 0x128, 0x7, 0x1b, 0x2, 0x2, 0x126, 0x129, 0x5, 0x32, 0x1a, 
    0x2, 0x127, 0x129, 0x5, 0x34, 0x1b, 0x2, 0x128, 0x126, 0x3, 0x2, 0x2, 
    0x2, 0x128, 0x127, 0x3, 0x2, 0x2, 0x2, 0x129, 0x12f, 0x3, 0x2, 0x2, 
    0x2, 0x12a, 0x12d, 0x7, 0x1a, 0x2, 0x2, 0x12b, 0x12e, 0x5, 0x32, 0x1a, 
    0x2, 0x12c, 0x12e, 0x5, 0x34, 0x1b, 0x2, 0x12d, 0x12b, 0x3, 0x2, 0x2, 
    0x2, 0x12d, 0x12c, 0x3, 0x2, 0x2, 0x2, 0x12e, 0x130, 0x3, 0x2, 0x2, 
    0x2, 0x12f, 0x12a, 0x3, 0x2, 0x2, 0x2, 0x130, 0x131, 0x3, 0x2, 0x2, 
    0x2, 0x131, 0x12f, 0x3, 0x2, 0x2, 0x2, 0x131, 0x132, 0x3, 0x2, 0x2, 
    0x2, 0x132, 0x2f, 0x3, 0x2, 0x2, 0x2, 0x133, 0x134, 0x5, 0x42, 0x22, 
    0x2, 0x134, 0x31, 0x3, 0x2, 0x2, 0x2, 0x135, 0x136, 0x5, 0x30, 0x19, 
    0x2, 0x136, 0x138, 0x7, 0x6, 0x2, 0x2, 0x137, 0x139, 0x5, 0x38, 0x1d, 
    0x2, 0x138, 0x137, 0x3, 0x2, 0x2, 0x2, 0x138, 0x139, 0x3, 0x2, 0x2, 
    0x2, 0x139, 0x13a, 0x3, 0x2, 0x2, 0x2, 0x13a, 0x13b, 0x7, 0x7, 0x2, 
    0x2, 0x13b, 0x13e, 0x3, 0x2, 0x2, 0x2, 0x13c, 0x13e, 0x5, 0x30, 0x19, 
    0x2, 0x13d, 0x135, 0x3, 0x2, 0x2, 0x2, 0x13d, 0x13c, 0x3, 0x2, 0x2, 
    0x2, 0x13e, 0x33, 0x3, 0x2, 0x2, 0x2, 0x13f, 0x140, 0x5, 0x30, 0x19, 
    0x2, 0x140, 0x141, 0x7, 0x1c, 0x2, 0x2, 0x141, 0x143, 0x7, 0x6, 0x2, 
    0x2, 0x142, 0x144, 0x5, 0x38, 0x1d, 0x2, 0x143, 0x142, 0x3, 0x2, 0x2, 
    0x2, 0x143, 0x144, 0x3, 0x2, 0x2, 0x2, 0x144, 0x145, 0x3, 0x2, 0x2, 
    0x2, 0x145, 0x146, 0x7, 0x7, 0x2, 0x2, 0x146, 0x35, 0x3, 0x2, 0x2, 0x2, 
    0x147, 0x14a, 0x5, 0x42, 0x22, 0x2, 0x148, 0x14a, 0x7, 0x24, 0x2, 0x2, 
    0x149, 0x147, 0x3, 0x2, 0x2, 0x2, 0x149, 0x148, 0x3, 0x2, 0x2, 0x2, 
    0x14a, 0x37, 0x3, 0x2, 0x2, 0x2, 0x14b, 0x14e, 0x5, 0x36, 0x1c, 0x2, 
    0x14c, 0x14d, 0x7, 0xf, 0x2, 0x2, 0x14d, 0x14f, 0x5, 0x36, 0x1c, 0x2, 
    0x14e, 0x14c, 0x3, 0x2, 0x2, 0x2, 0x14f, 0x150, 0x3, 0x2, 0x2, 0x2, 
    0x150, 0x151, 0x3, 0x2, 0x2, 0x2, 0x150, 0x14e, 0x3, 0x2, 0x2, 0x2, 
    0x151, 0x39, 0x3, 0x2, 0x2, 0x2, 0x152, 0x153, 0x7, 0x21, 0x2, 0x2, 
    0x153, 0x3b, 0x3, 0x2, 0x2, 0x2, 0x154, 0x157, 0x7, 0x1d, 0x2, 0x2, 
    0x155, 0x158, 0x7, 0x24, 0x2, 0x2, 0x156, 0x158, 0x5, 0x42, 0x22, 0x2, 
    0x157, 0x155, 0x3, 0x2, 0x2, 0x2, 0x157, 0x156, 0x3, 0x2, 0x2, 0x2, 
    0x158, 0x159, 0x3, 0x2, 0x2, 0x2, 0x159, 0x15c, 0x7, 0xc, 0x2, 0x2, 
    0x15a, 0x15d, 0x7, 0x24, 0x2, 0x2, 0x15b, 0x15d, 0x5, 0x42, 0x22, 0x2, 
    0x15c, 0x15a, 0x3, 0x2, 0x2, 0x2, 0x15c, 0x15b, 0x3, 0x2, 0x2, 0x2, 
    0x15d, 0x15e, 0x3, 0x2, 0x2, 0x2, 0x15e, 0x15f, 0x7, 0x1e, 0x2, 0x2, 
    0x15f, 0x3d, 0x3, 0x2, 0x2, 0x2, 0x160, 0x163, 0x5, 0x40, 0x21, 0x2, 
    0x161, 0x162, 0x7, 0xf, 0x2, 0x2, 0x162, 0x164, 0x5, 0x3e, 0x20, 0x2, 
    0x163, 0x161, 0x3, 0x2, 0x2, 0x2, 0x163, 0x164, 0x3, 0x2, 0x2, 0x2, 
    0x164, 0x3f, 0x3, 0x2, 0x2, 0x2, 0x165, 0x166, 0x5, 0x42, 0x22, 0x2, 
    0x166, 0x41, 0x3, 0x2, 0x2, 0x2, 0x167, 0x168, 0x7, 0x22, 0x2, 0x2, 
    0x168, 0x43, 0x3, 0x2, 0x2, 0x2, 0x169, 0x16a, 0x7, 0x1f, 0x2, 0x2, 
    0x16a, 0x16b, 0x5, 0x46, 0x24, 0x2, 0x16b, 0x16c, 0x7, 0xf, 0x2, 0x2, 
    0x16c, 0x16d, 0x5, 0x46, 0x24, 0x2, 0x16d, 0x16e, 0x7, 0x20, 0x2, 0x2, 
    0x16e, 0x45, 0x3, 0x2, 0x2, 0x2, 0x16f, 0x170, 0x7, 0x23, 0x2, 0x2, 
    0x170, 0x47, 0x3, 0x2, 0x2, 0x2, 0x171, 0x172, 0x7, 0x25, 0x2, 0x2, 
    0x172, 0x49, 0x3, 0x2, 0x2, 0x2, 0x2b, 0x4e, 0x58, 0x63, 0x68, 0x6b, 
    0x72, 0x7c, 0x80, 0x8c, 0x99, 0xa5, 0xb8, 0xd1, 0xd6, 0xd8, 0xda, 0xe5, 
    0xeb, 0xf1, 0xf3, 0xf9, 0xfe, 0x104, 0x106, 0x10c, 0x111, 0x116, 0x11a, 
    0x120, 0x122, 0x128, 0x12d, 0x131, 0x138, 0x13d, 0x143, 0x149, 0x150, 
    0x157, 0x15c, 0x163, 
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
