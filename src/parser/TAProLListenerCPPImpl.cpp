#include "TAProLListenerCPPImpl.hpp"

namespace exatn {

namespace parser {

void TAProLListenerCPPImpl::enterEntry(TAProLParser::EntryContext *ctx) {}

void TAProLListenerCPPImpl::enterScope(TAProLParser::ScopeContext *ctx) {
  cpp_source << "exatn::openScope(\"" << ctx->scopename(0)->getText() << "\");"
             << std::endl;
  return;
}

void TAProLListenerCPPImpl::exitScope(TAProLParser::ScopeContext *ctx) {
  cpp_source << "exatn::closeScope();" << std::endl;
  return;
}

void TAProLListenerCPPImpl::enterCode(TAProLParser::CodeContext *ctx) {}

void TAProLListenerCPPImpl::exitCode(TAProLParser::CodeContext *ctx) {}

void TAProLListenerCPPImpl::enterSpace(TAProLParser::SpaceContext *ctx) {
  for (auto space : ctx->spacedeflist()->spacedef()) {
    cpp_source << "auto _" << space->spacename()->getText()
               << " = exatn::createVectorSpace(\""
               << space->spacename()->getText() << "\",("
               << space->range()->upperbound()->getText() << " - "
               << space->range()->lowerbound()->getText() << " + 1));"
               << std::endl;
  }
  return;
}

void TAProLListenerCPPImpl::enterSubspace(TAProLParser::SubspaceContext *ctx) {
  for (auto subspace : ctx->spacedeflist()->spacedef()) {
    cpp_source << "auto _" << subspace->spacename()->getText()
               << " = exatn::createSubspace(\""
               << subspace->spacename()->getText() << "\",\""
               << ctx->spacename()->getText()
               << "\",std::pair<exatn::DimOffset, exatn::DimOffset>{"
               << subspace->range()->lowerbound()->getText() << ","
               << subspace->range()->upperbound()->getText() << "});"
               << std::endl;
    cpp_source << "auto _" << subspace->spacename()->getText() << "_space = "
               << "_" << ctx->spacename()->getText() << ";" << std::endl;
  }
  return;
}

void TAProLListenerCPPImpl::enterIndex(TAProLParser::IndexContext *ctx) {
  for (auto indx : ctx->indexlist()->indexname()) {
    cpp_source << "auto _" << indx->getText() << " = "
               << "_" << ctx->spacename()->getText() << ";" << std::endl;
    cpp_source << "auto _" << indx->getText() << "_space = "
               << "_" << ctx->spacename()->getText() << "_space;" << std::endl;
  }
  return;
}

void TAProLListenerCPPImpl::enterAssign(TAProLParser::AssignContext *ctx) {
  if (ctx->methodname() != nullptr) {
    cpp_source << "exatn::transformTensor(\""
               << ctx->tensor()->tensorname()->getText() << "\","
               << ctx->methodname()->getText() << ");" << std::endl;
  } else {
    std::string tensor_data_type = "exatn::TensorElementType::REAL64";
    if (ctx->complex() != nullptr)
      tensor_data_type = "exatn::TensorElementType::COMPLEX64";
    
    cpp_source << "exatn::createTensor(\""
               << ctx->tensor()->tensorname()->getText() << "\","
               << tensor_data_type << ",TensorSignature{";
    if (ctx->tensor()->indexlist() != nullptr) {
      const auto &indices = ctx->tensor()->indexlist()->indexname();
      for (auto indx = indices.cbegin(); indx != indices.cend(); ++indx) {
        cpp_source << "{"
                   << "_" << (*indx)->getText() << "_space"
                   << ","
                   << "_" << (*indx)->getText() << "}";
        if (std::distance(indx, indices.end()) > 1)
          cpp_source << ",";
      }
    }
    cpp_source << "});" << std::endl;
    if (ctx->datacontainer() != nullptr) {
      cpp_source << "exatn::initTensorData(\""
                 << ctx->tensor()->tensorname()->getText() << "\","
                 << ctx->datacontainer()->getText() << ");" << std::endl;
    } else {
      if (ctx->complex() != nullptr) {
        auto val_str = ctx->complex()->getText();
        val_str.erase(std::remove_if(val_str.begin(), val_str.end(),
                       [](char c) { return c == '{' || c == '}'; }), val_str.end());

        cpp_source << "exatn::initTensor(\""
                   << ctx->tensor()->tensorname()->getText()
                   << "\",std::complex<double>(" << val_str << "));"
                   << std::endl;
      } else {
        if (ctx->real() != nullptr) {
          cpp_source << "exatn::initTensor(\""
                     << ctx->tensor()->tensorname()->getText() << "\","
                     << ctx->real()->getText() << ");" << std::endl;
        }
      }
    }
  }
  return;
}

void TAProLListenerCPPImpl::enterRetrieve(TAProLParser::RetrieveContext *ctx) {
  const std::string &tensor_name = ctx->datacontainer()->getText();
  auto iter = args.find(tensor_name);
  if (iter == args.end()) {
    args.emplace(std::make_pair(tensor_name, "talsh::Tensor"));
    if (ctx->tensor() != nullptr) {
      cpp_source << "auto " << tensor_name << " = "
                 << "exatn::getLocalTensor(\""
                 << ctx->tensor()->tensorname()->getText() << "\");"
                 << std::endl;
    } else if (ctx->tensorname() != nullptr) {
      cpp_source << "auto " << tensor_name << " = "
                 << "exatn::getLocalTensor(\"" << ctx->tensorname()->getText()
                 << "\");" << std::endl;
    }
  } else {
    if (ctx->tensor() != nullptr) {
      cpp_source << tensor_name << " = "
                 << "exatn::getLocalTensor(\""
                 << ctx->tensor()->tensorname()->getText() << "\");"
                 << std::endl;
    } else if (ctx->tensorname() != nullptr) {
      cpp_source << tensor_name << " = "
                 << "exatn::getLocalTensor(\"" << ctx->tensorname()->getText()
                 << "\");" << std::endl;
    }
  }
  return;
}

void TAProLListenerCPPImpl::enterLoad(TAProLParser::LoadContext *ctx) {}

void TAProLListenerCPPImpl::enterSave(TAProLParser::SaveContext *ctx) {}

void TAProLListenerCPPImpl::enterDestroy(TAProLParser::DestroyContext *ctx) {
  if (ctx->tensorlist() != nullptr) {
    for (auto tens : ctx->tensorlist()->tensor()) {
      cpp_source << "exatn::destroyTensor(\"" << tens->tensorname()->getText()
                 << "\");" << std::endl;
    }
    for (auto tens : ctx->tensorlist()->tensorname()) {
      cpp_source << "exatn::destroyTensor(\"" << tens->getText() << "\");"
                 << std::endl;
    }
  } else {
    if (ctx->tensor() != nullptr) {
      cpp_source << "exatn::destroyTensor(\""
                 << ctx->tensor()->tensorname()->getText() << "\");"
                 << std::endl;
    } else {
      if (ctx->tensorname() != nullptr) {
        cpp_source << "exatn::destroyTensor(\"" << ctx->tensorname()->getText()
                   << "\");" << std::endl;
      }
    }
  }
  return;
}

void TAProLListenerCPPImpl::enterNorm1(TAProLParser::Norm1Context *ctx) {
  const std::string &scalar_name = ctx->scalar()->getText();
  auto iter = args.find(scalar_name);
  if (iter == args.end()) {
    args.emplace(std::make_pair(scalar_name, "double"));
    cpp_source << "double " << scalar_name << ";" << std::endl;
  }
  if (ctx->tensor() != nullptr) {
    cpp_source << "exatn::computeNorm1Sync(\""
               << ctx->tensor()->tensorname()->getText() << "\"," << scalar_name
               << ");" << std::endl;
  } else if (ctx->tensorname() != nullptr) {
    cpp_source << "exatn::computeNorm1Sync(\"" << ctx->tensorname()->getText()
               << "\"," << scalar_name << ");" << std::endl;
  }
  return;
}

void TAProLListenerCPPImpl::enterNorm2(TAProLParser::Norm2Context *ctx) {
  const std::string &scalar_name = ctx->scalar()->getText();
  auto iter = args.find(scalar_name);
  if (iter == args.end()) {
    args.emplace(std::make_pair(scalar_name, "double"));
    cpp_source << "double " << scalar_name << ";" << std::endl;
  }
  if (ctx->tensor() != nullptr) {
    cpp_source << "exatn::computeNorm2Sync(\""
               << ctx->tensor()->tensorname()->getText() << "\"," << scalar_name
               << ");" << std::endl;
  } else if (ctx->tensorname() != nullptr) {
    cpp_source << "exatn::computeNorm2Sync(\"" << ctx->tensorname()->getText()
               << "\"," << scalar_name << ");" << std::endl;
  }
  return;
}

void TAProLListenerCPPImpl::enterMaxabs(TAProLParser::MaxabsContext *ctx) {
  const std::string &scalar_name = ctx->scalar()->getText();
  auto iter = args.find(scalar_name);
  if (iter == args.end()) {
    args.emplace(std::make_pair(scalar_name, "double"));
    cpp_source << "double " << scalar_name << ";" << std::endl;
  }
  if (ctx->tensor() != nullptr) {
    cpp_source << "exatn::computeMaxabs(\""
               << ctx->tensor()->tensorname()->getText() << "\"," << scalar_name
               << ");" << std::endl;
  } else if (ctx->tensorname() != nullptr) {
    cpp_source << "exatn::computeMaxabs(\"" << ctx->tensorname()->getText()
               << "\"," << scalar_name << ");" << std::endl;
  }
  return;
}

void TAProLListenerCPPImpl::enterScale(TAProLParser::ScaleContext *ctx) {
  cpp_source << "exatn::scaleTensor(\""
             << ctx->tensor()->tensorname()->getText() << "\","
             << ctx->prefactor()->getText() << ");" << std::endl;
  return;
}

void TAProLListenerCPPImpl::enterCopy(TAProLParser::CopyContext *ctx) {}

void TAProLListenerCPPImpl::enterAddition(TAProLParser::AdditionContext *ctx) {
  if (ctx->prefactor() != nullptr) {
    cpp_source << "exatn::addTensors(\"" << ctx->getText() << "\","
               << ctx->prefactor()->getText() << ");" << std::endl;
  } else {
    cpp_source << "exatn::addTensors(\"" << ctx->getText() << "\", 1.0);"
               << std::endl;
  }
  return;
}

void TAProLListenerCPPImpl::enterContraction(
    TAProLParser::ContractionContext *ctx) {
  if (ctx->prefactor() != nullptr) {
    cpp_source << "exatn::contractTensors(\"" << ctx->getText() << "\","
               << ctx->prefactor()->getText() << ");" << std::endl;
  } else {
    cpp_source << "exatn::contractTensors(\"" << ctx->getText() << "\", 1.0);"
               << std::endl;
  }
  return;
}

void TAProLListenerCPPImpl::enterCompositeproduct(
    TAProLParser::CompositeproductContext *ctx) {
  cpp_source << "exatn::evaluateTensorNetwork(\""
             << "_SmokyTN"
             << "\",\"" << ctx->getText() << "\");" << std::endl;
  return;
}

void TAProLListenerCPPImpl::enterTensornetwork(
    TAProLParser::TensornetworkContext *ctx) {}

} // namespace parser

} // namespace exatn
