#include "TAProLListener.hpp"

namespace exatn{

namespace parser {


void TAProLListener::enterEntry(TAProLParser::EntryContext * ctx) { }
void TAProLListener::exitEntry(TAProLParser::EntryContext * ctx) { }

void TAProLListener::enterScope(TAProLParser::ScopeContext * ctx) { }
void TAProLListener::exitScope(TAProLParser::ScopeContext * ctx) { }

void TAProLListener::enterSimpleop(TAProLParser::SimpleopContext * ctx) { }
void TAProLListener::exitSimpleop(TAProLParser::SimpleopContext * ctx) { }

void TAProLListener::enterCompositeop(TAProLParser::CompositeopContext * ctx) { }
void TAProLListener::exitCompositeop(TAProLParser::CompositeopContext * ctx) { }

void TAProLListener::enterSpace(TAProLParser::SpaceContext * ctx) { }
void TAProLListener::exitSpace(TAProLParser::SpaceContext * ctx) { }

void TAProLListener::enterSubspace(TAProLParser::SubspaceContext * ctx) { }
void TAProLListener::exitSubspace(TAProLParser::SubspaceContext * ctx) { }

void TAProLListener::enterIndex(TAProLParser::IndexContext * ctx) { }
void TAProLListener::exitIndex(TAProLParser::IndexContext * ctx) { }

void TAProLListener::enterAssignment(TAProLParser::AssignmentContext * ctx) { }
void TAProLListener::exitAssignment(TAProLParser::AssignmentContext * ctx) { }

void TAProLListener::enterLoad(TAProLParser::LoadContext * ctx) { }
void TAProLListener::exitLoad(TAProLParser::LoadContext * ctx) { }

void TAProLListener::enterSave(TAProLParser::SaveContext * ctx) { }
void TAProLListener::exitSave(TAProLParser::SaveContext * ctx) { }

void TAProLListener::enterDestroy(TAProLParser::DestroyContext * ctx) { }
void TAProLListener::exitDestroy(TAProLParser::DestroyContext * ctx) { }

void TAProLListener::enterCopy(TAProLParser::CopyContext * ctx) { }
void TAProLListener::exitCopy(TAProLParser::CopyContext * ctx) { }

void TAProLListener::enterScale(TAProLParser::ScaleContext * ctx) { }
void TAProLListener::exitScale(TAProLParser::ScaleContext * ctx) { }

void TAProLListener::enterUnaryop(TAProLParser::UnaryopContext * ctx) { }
void TAProLListener::exitUnaryop(TAProLParser::UnaryopContext * ctx) { }

void TAProLListener::enterBinaryop(TAProLParser::BinaryopContext * ctx) { }
void TAProLListener::exitBinaryop(TAProLParser::BinaryopContext * ctx) { }

void TAProLListener::enterCompositeproduct(TAProLParser::CompositeproductContext * ctx) { }
void TAProLListener::exitCompositeproduct(TAProLParser::CompositeproductContext * ctx) { }

void TAProLListener::enterTensornetwork(TAProLParser::TensornetworkContext * ctx) { }
void TAProLListener::exitTensornetwork(TAProLParser::TensornetworkContext * ctx) { }
}
}