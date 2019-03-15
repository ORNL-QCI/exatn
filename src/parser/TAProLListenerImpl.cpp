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
    std::cout << "ENTERING SCOPE: " << ctx->scopebeginname->getText() << ", " << ctx->scopeendname->getText() << "\n";
    // check that scope end name and scope begin name are equal
}

void TAProLListenerImpl::enterSpace(TAProLParser::SpaceContext *ctx) {}

void TAProLListenerImpl::enterSubspace(TAProLParser::SubspaceContext *ctx) {
    std::cout << "Entering subsbace: " << ctx->getText() << "\n";
    if (ctx->spacename != nullptr) {
      auto spacename = ctx->spacename->getText();
    }

    std::cout << ctx->spacelist()->spacename->getText() << "\n";
}

void TAProLListenerImpl::enterIndex(TAProLParser::IndexContext *ctx) {
  auto subspace = ctx->subspacename->getText();

  int nIndices = ctx->idx().size();

  std::vector<std::string> indices;
  for (int i = 0; i < nIndices; i++) {
    std::cout << "adding index: " << ctx->idx(i)->getText() << "\n";
    indices.push_back(ctx->idx(i)->getText());
  }
}

void TAProLListenerImpl::enterAssignment(TAProLParser::AssignmentContext *ctx) {
    if (ctx->complex() != nullptr) {
        // this is an assigment of the form
        // X2 = {0.0,0.0} or T2 = {1.0,0.0}
        std::cout << "= assignment: " << ctx->getText() << "\n";

    } else if (ctx->string() != nullptr) {
        // this is a method assigment
        std::cout << "method assignment: " << ctx->getText() << "\n";


    }
}

void TAProLListenerImpl::enterLoad(TAProLParser::LoadContext *ctx) {}

void TAProLListenerImpl::enterSave(TAProLParser::SaveContext *ctx) {
    auto tensor = ctx->tensor()->getText();
    auto tag = ctx->string()->getText();
    tag = tag.substr(1,tag.length()-2);
    std::cout << "Entering Save: " << ctx->getText() << "\n";
    std::cout << "Save Data: " << tensor << "\n";
    std::cout << "Save Data: " << tag << "\n";
}

void TAProLListenerImpl::enterDestroy(TAProLParser::DestroyContext *ctx) {
    std::cout << "Destroying Tensor: " << ctx->getText() << "\n";
    
}

void TAProLListenerImpl::enterCopy(TAProLParser::CopyContext *ctx) {}

void TAProLListenerImpl::enterScale(TAProLParser::ScaleContext *ctx) {}

void TAProLListenerImpl::enterUnaryop(TAProLParser::UnaryopContext *ctx) {
    std::cout << "Entering unary op: " << ctx->getText() << "\n";
}

void TAProLListenerImpl::enterBinaryop(TAProLParser::BinaryopContext *ctx) {
    std::cout << "Entering binary op: " << ctx->getText() << "\n";
}

void TAProLListenerImpl::enterCompositeproduct(
    TAProLParser::CompositeproductContext *ctx) {}

void TAProLListenerImpl::enterTensornetwork(
    TAProLParser::TensornetworkContext *ctx) {}
} // namespace parser
} // namespace exatn