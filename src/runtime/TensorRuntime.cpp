#include "TensorRuntime.hpp"
#include "exatn.hpp"

namespace exatn {
namespace runtime {

void TensorRuntime::openScope(const std::string & scopeName) {
  assert(scopeName.length() > 0);
  // Complete the current scope first:
  if(currentScope_.length() > 0) closeScope();
  // Create new graph with name given by scope name and store it in the dags map:
  auto new_dag = dags_.emplace(std::make_pair(
                                scopeName,
                                exatn::getService<TensorGraph>("boost-digraph")
                               )
                              );
  assert(new_dag.second); // to make sure there was no other scope with the same name
  currentDag_ = (*(new_dag.first)).second.get();
  currentScope_ = scopeName; // change the name of the current scope
  return;
}


void TensorRuntime::pauseScope() {
  //`complete all currently executed tensor operations
  return;
}


void TensorRuntime::resumeScope(const std::string & scopeName) {
  assert(scopeName.length() > 0);
  // Pause the current scope first:
  if(currentScope_.length() > 0) pauseScope();
  //`resume the execution of the previously paused scope
  currentDag_ = dags_[scopeName].get();
  currentScope_ = scopeName; // change the name of the current scope
  return;
}


void TensorRuntime::closeScope() {
  if(currentScope_.length() > 0){
    //`complete all operations in the current scope:
    assert(dags_.erase(currentScope_) == 1);
    currentDag_ = nullptr;
    currentScope_ = "";
  }
  return;
}


VertexIdType TensorRuntime::submit(std::shared_ptr<numerics::TensorOperation> op) {
  assert(currentScope_.length() > 0);
  auto newop_outid = op->getTensorOperandHash(0);
  currentDag_->lock();
  
  currentDag_->unlock();
  return 0; //???
}


bool TensorRuntime::sync(const numerics::TensorOperation & op, bool wait) {
  assert(currentScope_.length() > 0);
  bool completed = false;
  const auto op_outid = op.getTensorOperandHash(0);
  currentDag_->lock();
  
  currentDag_->unlock();
  return completed;
}

bool TensorRuntime::sync(const numerics::Tensor & tensor, bool wait) {
  assert(currentScope_.length() > 0);
  bool completed = false;
  const auto op_outid = tensor.getTensorHash();
  currentDag_->lock();
  
  currentDag_->unlock();
  return completed;
}

TensorDenseBlock TensorRuntime::getTensorData(const numerics::Tensor & tensor) {
  // sync
  assert(sync(tensor,true));
  // get tensor data after sync
  return TensorDenseBlock();
}

} // namespace runtime
} // namespace exatn
