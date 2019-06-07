#include "TensorRuntime.hpp"

namespace exatn {
namespace runtime {
void TensorRuntime::openScope(const std::string &scopeName) {
  currentScope = scopeName;
  // create new graph with name given by scope name
  TensorGraph tg;
  // store it in the dags map
  dags[currentScope]=tg; 
  // save currentScope name
}

void TensorRuntime::closeScope() { currentScope = ""; }

void TensorRuntime::submit(TensorOp &op) {
  // add on to the graph
  // work on graph at dags[currentScope]
}

void TensorRuntime::sync(const exatn::numerics::Tensor &tensor) {
  // sync on a particular tensor, everything related to tensor
  // must complete
}

TensorDenseBlock
TensorRuntime::getTensorData(const exatn::numerics::Tensor &tensor) {
  // get tensor data after sync
  return TensorDenseBlock();
}
} // namespace runtime
} // namespace exatn
