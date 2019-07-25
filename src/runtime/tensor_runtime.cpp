#include "tensor_runtime.hpp"
#include "exatn.hpp"

namespace exatn {
namespace runtime {

TensorRuntime::TensorRuntime():
 current_dag_(nullptr), executing_(false), alive_(false)
{
}


TensorRuntime::~TensorRuntime()
{
  if(alive_.load()){
    alive_ = false; //signal to the execution thread to finish
    exec_thread_.join(); //wait until the execution thread has finished
  }
}


void TensorRuntime::launchExecutionThread()
{
  alive_ = true;
  exec_thread_ = std::thread(&TensorRuntime::executionThreadWorkflow,this);
  return;
}


void TensorRuntime::executionThreadWorkflow()
{
  //`Implement
  bool finished = !(alive_.load());
  if(finished) return;
}


void TensorRuntime::openScope(const std::string & scopeName) {
  assert(scopeName.length() > 0);
  // Complete the current scope first:
  if(current_scope_.length() > 0) closeScope();
  // Create new graph with name given by scope name and store it in the dags map:
  auto new_dag = dags_.emplace(std::make_pair(
                                scopeName,
                                exatn::getService<TensorGraph>("boost-digraph")
                               )
                              );
  assert(new_dag.second); // to make sure there was no other scope with the same name
  current_dag_ = (*(new_dag.first)).second.get();
  current_scope_ = scopeName; // change the name of the current scope
  return;
}


void TensorRuntime::pauseScope() {
  //`complete all currently executed tensor operations
  return;
}


void TensorRuntime::resumeScope(const std::string & scopeName) {
  assert(scopeName.length() > 0);
  // Pause the current scope first:
  if(current_scope_.length() > 0) pauseScope();
  //`resume the execution of the previously paused scope
  current_dag_ = dags_[scopeName].get();
  current_scope_ = scopeName; // change the name of the current scope
  return;
}


void TensorRuntime::closeScope() {
  if(current_scope_.length() > 0){
    //`complete all operations in the current scope:
    assert(dags_.erase(current_scope_) == 1);
    current_dag_ = nullptr;
    current_scope_ = "";
  }
  return;
}


VertexIdType TensorRuntime::submit(std::shared_ptr<TensorOperation> op) {
  assert(current_scope_.length() > 0);
  auto newop_outid = op->getTensorOperandHash(0);
  
  return 0; //???
}


bool TensorRuntime::sync(const TensorOperation & op, bool wait) {
  assert(current_scope_.length() > 0);
  bool completed = false;
  const auto op_outid = op.getTensorOperandHash(0);
  
  return completed;
}

bool TensorRuntime::sync(const Tensor & tensor, bool wait) {
  assert(current_scope_.length() > 0);
  bool completed = false;
  const auto op_outid = tensor.getTensorHash();
  
  return completed;
}

TensorDenseBlock TensorRuntime::getTensorData(const Tensor & tensor) {
  // sync
  assert(sync(tensor,true));
  // get tensor data after sync
  return TensorDenseBlock();
}

} // namespace runtime
} // namespace exatn
