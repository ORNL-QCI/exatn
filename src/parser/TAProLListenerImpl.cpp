#include "TAProLListenerImpl.hpp"

namespace exatn {

namespace parser {

/*
entry: main # First function to start from, hence ENTRY
scope main group() # Function definition
 subspace(): s0=[0:127]
 index(s0): a,b,c,d,i,j,k,l
 H2(a,b,c,d) = method("HamiltonianTest")
 T2(a,b,c,d) = {1.0,0.0}
 Z2(a,b,c,d) = {0.0,0.0}
 Z2(a,b,c,d) += H2(i,j,k,l) * T2(c,d,i,j) * T2(a,b,k,l)
 X2() = {0.0,0.0}
 X2() += Z2+(a,b,c,d) * Z2(a,b,c,d)
 save X2: tag("Z2_norm")
 ~X2
 ~Z2
 ~T2
 ~H2
end scope main
*/

void TAProLListenerImpl::enterEntry(TAProLParser::EntryContext *ctx) {
  entryValue = ctx->ID()->getText();
  std::cout << "ENTERING ENTRY " << entryValue << "\n";
}

void TAProLListenerImpl::enterScope(TAProLParser::ScopeContext *ctx) {
    std::cout << "ENTERING SCOPE: " << ctx->getText() << "\n";
}

void TAProLListenerImpl::enterSimpleop(TAProLParser::SimpleopContext *ctx) {}

void TAProLListenerImpl::enterCompositeop(
    TAProLParser::CompositeopContext *ctx) {}

void TAProLListenerImpl::enterSpace(TAProLParser::SpaceContext *ctx) {}

void TAProLListenerImpl::enterSubspace(TAProLParser::SubspaceContext *ctx) {}

void TAProLListenerImpl::enterIndex(TAProLParser::IndexContext *ctx) {
  std::cout << "ENTERING INDEX\n";
  auto subspace = ctx->subspacename->getText();
  int nIndices = ctx->id().size();
  std::vector<std::string> indices;
  for (int i = 0; i < nIndices; i++) {
    std::cout << "ADDING INDEX: " << ctx->id(i)->getText() << "\n";
    indices.push_back(ctx->id(i)->getText());
  }
}

void TAProLListenerImpl::enterAssignment(TAProLParser::AssignmentContext *ctx) {
}

void TAProLListenerImpl::enterLoad(TAProLParser::LoadContext *ctx) {}

void TAProLListenerImpl::enterSave(TAProLParser::SaveContext *ctx) {}

void TAProLListenerImpl::enterDestroy(TAProLParser::DestroyContext *ctx) {}

void TAProLListenerImpl::enterCopy(TAProLParser::CopyContext *ctx) {}

void TAProLListenerImpl::enterScale(TAProLParser::ScaleContext *ctx) {}

void TAProLListenerImpl::enterUnaryop(TAProLParser::UnaryopContext *ctx) {}

void TAProLListenerImpl::enterBinaryop(TAProLParser::BinaryopContext *ctx) {}

void TAProLListenerImpl::enterCompositeproduct(
    TAProLParser::CompositeproductContext *ctx) {}

void TAProLListenerImpl::enterTensornetwork(
    TAProLParser::TensornetworkContext *ctx) {}
} // namespace parser
} // namespace exatn