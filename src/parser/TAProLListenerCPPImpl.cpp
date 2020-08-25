#include "TAProLListenerCPPImpl.hpp"

#include <string>

namespace exatn {

namespace parser {

void TAProLListenerCPPImpl::enterEntry(TAProLParser::EntryContext * ctx)
{
}

void TAProLListenerCPPImpl::enterScope(TAProLParser::ScopeContext * ctx)
{
 cpp_source << "exatn::openScope(\"" << ctx->scopename(0)->getText() << "\");" << std::endl;
 return;
}

void TAProLListenerCPPImpl::exitScope(TAProLParser::ScopeContext * ctx)
{
 cpp_source << "exatn::closeScope();" << std::endl;
 return;
}

void TAProLListenerCPPImpl::enterCode(TAProLParser::CodeContext * ctx)
{
}

void TAProLListenerCPPImpl::exitCode(TAProLParser::CodeContext * ctx)
{
}

void TAProLListenerCPPImpl::enterSpace(TAProLParser::SpaceContext * ctx)
{
 for(auto space: ctx->spacedeflist()->spacedef()){
  cpp_source << "auto _" << space->spacename()->getText()
             << " = exatn::createVectorSpace(\"" << space->spacename()->getText()
             << "\",(" << space->range()->upperbound()->getText() << " - "
             << space->range()->lowerbound()->getText() << " + 1));" << std::endl;
 }
 return;
}

void TAProLListenerCPPImpl::enterSubspace(TAProLParser::SubspaceContext * ctx)
{
 for(auto subspace: ctx->spacedeflist()->spacedef()){
  cpp_source << "auto _" << subspace->spacename()->getText()
             << " = exatn::createSubspace(\"" << subspace->spacename()->getText()
             << "\",\"" << ctx->spacename()->getText()
             << "\",std::pair<exatn::DimOffset>{"
             << subspace->range()->lowerbound()->getText() << ","
             << subspace->range()->upperbound()->getText() << "});" << std::endl;
  cpp_source << "auto _" << subspace->spacename()->getText() << "_space = "
             << "_" << ctx->spacename()->getText() << ";" << std::endl;
 }
 return;
}

void TAProLListenerCPPImpl::enterIndex(TAProLParser::IndexContext * ctx)
{
 for(auto indx: ctx->indexlist()->indexname()){
  cpp_source << "auto _" << indx->getText() << " = "
             << "_" << ctx->spacename()->getText() << ";" << std::endl;
  cpp_source << "auto _" << indx->getText() << "_space = "
             << "_" << ctx->spacename()->getText() << "_space;" << std::endl;
 }
 return;
}

void TAProLListenerCPPImpl::enterAssign(TAProLParser::AssignContext * ctx)
{
 if(ctx->methodname() != nullptr){
  cpp_source << "exatn::transformTensor(\"" << ctx->tensor()->tensorname()->getText()
             << "\"," << ctx->methodname()->getText() << ");" << std::endl;
 }else{
  std::string tensor_data_type = "exatn::TensorElementType::REAL64";
  if(ctx->complex() != nullptr) tensor_data_type = "exatn::TensorElementType::COMPLEX64";
  cpp_source << "exatn::createTensor(\"" << ctx->tensor()->tensorname()->getText()
             << "\"," << tensor_data_type << ",TensorSignature{";
  if(ctx->tensor()->indexlist() != nullptr){
   const auto & indices = ctx->tensor()->indexlist()->indexname();
   for(auto indx = indices.cbegin(); indx != indices.cend(); ++indx){
    cpp_source << "{" << "_" << (*indx)->getText() << "_space" << "," << "_" << (*indx)->getText() << "}";
    if(std::distance(indx,indices.end()) > 1) cpp_source << ",";
   }
  }
  cpp_source << "});" << std::endl;
  if(ctx->complex() != nullptr){
   cpp_source << "exatn::initTensor(\"" << ctx->tensor()->tensorname()->getText()
              << "\",std::complex<double>(" << ctx->complex()->getText() << "));" << std::endl;
  }else{
   if(ctx->real() != nullptr){
    cpp_source << "exatn::initTensor(\"" << ctx->tensor()->tensorname()->getText()
               << "\"," << ctx->real()->getText() << ");" << std::endl;
   }
  }
 }
 return;
}

void TAProLListenerCPPImpl::enterLoad(TAProLParser::LoadContext * ctx)
{
}

void TAProLListenerCPPImpl::enterSave(TAProLParser::SaveContext * ctx)
{
}

void TAProLListenerCPPImpl::enterDestroy(TAProLParser::DestroyContext * ctx)
{
 if(ctx->tensorlist() != nullptr){
  for(auto tens: ctx->tensorlist()->tensor()){
   cpp_source << "exatn::destroyTensor(\"" << tens->tensorname()->getText() << "\");" << std::endl;
  }
  for(auto tens: ctx->tensorlist()->tensorname()){
   cpp_source << "exatn::destroyTensor(\"" << tens->getText() << "\");" << std::endl;
  }
 }else{
  if(ctx->tensor() != nullptr){
   cpp_source << "exatn::destroyTensor(\"" << ctx->tensor()->tensorname()->getText() << "\");" << std::endl;
  }else{
   if(ctx->tensorname() != nullptr){
    cpp_source << "exatn::destroyTensor(\"" << ctx->tensorname()->getText() << "\");" << std::endl;
   }
  }
 }
 return;
}

void TAProLListenerCPPImpl::enterNorm(TAProLParser::NormContext * ctx)
{
}

void TAProLListenerCPPImpl::enterScale(TAProLParser::ScaleContext * ctx)
{
}

void TAProLListenerCPPImpl::enterCopy(TAProLParser::CopyContext * ctx)
{
}

void TAProLListenerCPPImpl::enterUnaryop(TAProLParser::UnaryopContext * ctx)
{
}

void TAProLListenerCPPImpl::enterBinaryop(TAProLParser::BinaryopContext * ctx)
{
}

void TAProLListenerCPPImpl::enterCompositeproduct(TAProLParser::CompositeproductContext * ctx)
{
}

void TAProLListenerCPPImpl::enterTensornetwork(TAProLParser::TensornetworkContext * ctx)
{
}

} // namespace parser

} // namespace exatn
