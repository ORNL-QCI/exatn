#include "tensor_runtime.hpp"
#include "exatn_service.hpp"

namespace exatn {
namespace runtime {

TensorRuntime::TensorRuntime(const std::string & graph_executor_name, const std::string & node_executor_name):
 current_dag_(nullptr), executing_(false), alive_(false)
{
  graph_executor_ = exatn::getService<TensorGraphExecutor>(graph_executor_name);
  graph_executor_->resetNodeExecutor(exatn::getService<TensorNodeExecutor>(node_executor_name));
}


TensorRuntime::~TensorRuntime()
{
  if(alive_.load()){
    alive_.store(false); //signal for the execution thread to finish
    exec_thread_.join(); //wait until the execution thread has finished
  }
}


void TensorRuntime::launchExecutionThread()
{
  if(!(alive_.load())){
    alive_.store(true);
    exec_thread_ = std::thread(&TensorRuntime::executionThreadWorkflow,this);
  }
  return; //only the main thread returns to the client
}


void TensorRuntime::executionThreadWorkflow()
{
  while(alive_.load()){
    if(executing_.load()){ //executing_ is set to TRUE by the main thread when new operations are submitted
      graph_executor_->execute(*current_dag_);
      executing_.store(false); //executing_ is set to FALSE by the execution thread
    }
  }
  return; //end of execution thread life
}


void TensorRuntime::openScope(const std::string & scope_name) {
  assert(!scope_name.empty());
  // Complete the current scope first:
  if(currentScopeIsSet()){
    assert(scope_name != current_scope_);
    closeScope();
  }
  // Create new DAG with name given by scope name and store it in the dags map:
  auto new_dag = dags_.emplace(std::make_pair(
                                scope_name,
                                exatn::getService<TensorGraph>("boost-digraph")
                               )
                              );
  assert(new_dag.second); // make sure there was no other scope with the same name
  current_dag_ = (new_dag.first)->second.get(); //storing a non-owning raw pointer to the DAG
  current_scope_ = scope_name; // change the name of the current scope
  return;
}


void TensorRuntime::pauseScope() {
  graph_executor_->stopExecution(); //execution thread will pause and reset executing_ to FALSE
  return;
}


void TensorRuntime::resumeScope(const std::string & scope_name) {
  assert(!scope_name.empty());
  // Pause the current scope first:
  if(currentScopeIsSet()) pauseScope();
  while(executing_.load()){}; //wait until the execution thread stops executing previous DAG
  current_dag_ = dags_[scope_name].get(); //storing a non-owning raw pointer to the DAG
  current_scope_ = scope_name; // change the name of the current scope
  executing_.store(true); //will trigger DAG execution by the execution thread
  return;
}


void TensorRuntime::closeScope() {
  if(currentScopeIsSet()){
    const std::string scope_name = current_scope_;
    while(executing_.load()){}; //wait until the execution thread has completed execution of the current DAG
    current_scope_ = "";
    current_dag_ = nullptr;
    auto num_deleted = dags_.erase(scope_name);
    assert(num_deleted == 1);
  }
  return;
}


VertexIdType TensorRuntime::submit(std::shared_ptr<TensorOperation> op) {
  assert(currentScopeIsSet());
  auto node_id = current_dag_->addOperation(op);
  op->setId(node_id);
  executing_.store(true); //signal to the execution thread to execute the DAG
  return node_id;
}


bool TensorRuntime::sync(TensorOperation & op, bool wait) {
  assert(currentScopeIsSet());
  executing_.store(true); //reactivate the execution thread to execute the DAG in case it was not active
  auto opid = op.getId();
  bool completed = current_dag_->nodeExecuted(opid);
  while(wait && (!completed)) completed = current_dag_->nodeExecuted(opid);
  return completed;
}


bool TensorRuntime::sync(const Tensor & tensor, bool wait) {
  assert(currentScopeIsSet());
  executing_.store(true); //reactivate the execution thread to execute the DAG in case it was not active
  bool completed = (current_dag_->getTensorUpdateCount(tensor) == 0);
  while(wait && (!completed)) completed = (current_dag_->getTensorUpdateCount(tensor) == 0);
  return completed;
}


TensorDenseBlock TensorRuntime::getTensorData(const Tensor & tensor) {
  // Complete all submitted update operations on the tensor
  auto synced = sync(tensor,true);
  assert(synced);
  //`Get access to tensor data
  return TensorDenseBlock();
}

} // namespace runtime
} // namespace exatn
