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

void TAProLListenerImpl::enterEntry(TAProLParser::EntryContext *ctx){
 entryScopeName = ctx->id()->getText();
 std::cout << "ENTRY: " << entryScopeName << std::endl;
}

void TAProLListenerImpl::enterScope(TAProLParser::ScopeContext *ctx){
 std::cout << "SCOPE: " << ctx->id()->getText() << std::endl;
}

void TAProLListenerImpl::enterSpace(TAProLParser::SpaceContext *ctx){
 std::cout << "SPACES: ";
 for(const auto & spacedef: ctx->spacedeflist()->spacedef()) std::cout << spacedef->spacename()->id()->getText() << ",";
 std::cout << std::endl;
}

void TAProLListenerImpl::enterSubspace(TAProLParser::SubspaceContext *ctx){
 if(ctx->spacename() != nullptr){
  std::cout << "SUBSPACES of SPACE " << ctx->spacename()->getText() << ":";
  for(const auto & spacedef: ctx->spacedeflist()->spacedef()) std::cout << spacedef->spacename()->id()->getText() << ",";
  std::cout << std::endl;
 }
}

void TAProLListenerImpl::enterIndex(TAProLParser::IndexContext *ctx){
 std::cout << "INDICES: ";
 for(const auto & index: ctx->indexlist()->indexname()) std::cout << index->id()->getText() << ",";
 std::cout << std::endl;
}

void TAProLListenerImpl::enterAssignment(TAProLParser::AssignmentContext *ctx) {
 if(ctx->complex() != nullptr || ctx->real() != nullptr){
  // This is an assignment of the form
  // X2(a,b) = 0.0 or T2(a,b) = {1.0,0.0}
  std::cout << "ASSIGNMENT(=): " << ctx->getText() << std::endl;
 }else if(ctx->methodname() != nullptr){
  // This is an external method based assigment
  std::cout << "ASSIGNMENT(methof): " << ctx->getText() << std::endl;
 }
}

void TAProLListenerImpl::enterLoad(TAProLParser::LoadContext *ctx){

}

void TAProLListenerImpl::enterSave(TAProLParser::SaveContext *ctx) {
 auto tensor = ctx->tensor()->getText();
 auto tag = ctx->tagname()->getText();
 std::cout << "SAVE TENSOR    : " << ctx->getText() << std::endl;
 std::cout << " Saved tensor  : " << tensor << std::endl;
 std::cout << " Saved tag name: " << tag << std::endl;
}

void TAProLListenerImpl::enterDestroy(TAProLParser::DestroyContext *ctx){

}

void TAProLListenerImpl::enterCopy(TAProLParser::CopyContext *ctx){

}

void TAProLListenerImpl::enterScale(TAProLParser::ScaleContext *ctx){

}

void TAProLListenerImpl::enterUnaryop(TAProLParser::UnaryopContext *ctx){

}

void TAProLListenerImpl::enterBinaryop(TAProLParser::BinaryopContext *ctx){

}

void TAProLListenerImpl::enterCompositeproduct(TAProLParser::CompositeproductContext *ctx){

}

void TAProLListenerImpl::enterTensornetwork(TAProLParser::TensornetworkContext *ctx){

}

} // namespace parser

} // namespace exatn
