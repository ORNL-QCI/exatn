/** ExaTN:: Tensor Runtime: Task-based execution layer for tensor operations
REVISION: 2022/03/30

Copyright (C) 2018-2022 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
**/

#include "tensor_runtime.hpp"
#include "exatn_service.hpp"

#include "talshxx.hpp"

#ifdef MPI_ENABLED
#include "mpi.h"
#endif

#include <vector>
#include <iostream>

#include "errors.hpp"

//#define DEBUG

namespace exatn {
namespace runtime {

#ifdef MPI_ENABLED
static MPI_Comm global_mpi_comm; //MPI communicator used to initialize the tensor runtime

TensorRuntime::TensorRuntime(const MPICommProxy & communicator,
                             const ParamConf & parameters,
                             const std::string & graph_executor_name,
                             const std::string & node_executor_name):
 parameters_(parameters),
 graph_executor_name_(graph_executor_name), node_executor_name_(node_executor_name), current_dag_(nullptr),
 logging_(0), backend_(CompBackend::None), executing_(false), scope_set_(false), alive_(false)
{
#ifdef DEBUG
  const bool debugging = true;
#else
  const bool debugging = false;
#endif
  global_mpi_comm = *(communicator.get<MPI_Comm>());
  int mpi_error = MPI_Comm_size(global_mpi_comm,&num_processes_); assert(mpi_error == MPI_SUCCESS);
  mpi_error = MPI_Comm_rank(global_mpi_comm,&process_rank_); assert(mpi_error == MPI_SUCCESS);
  mpi_error = MPI_Comm_rank(MPI_COMM_WORLD,&global_process_rank_); assert(mpi_error == MPI_SUCCESS);
  graph_executor_ = exatn::getService<TensorGraphExecutor>(graph_executor_name_);
  if(debugging) std::cout << "#DEBUG(exatn::runtime::TensorRuntime)[MAIN_THREAD:Process " << process_rank_
                          << "]: DAG executor set to " << graph_executor_name_ << " + "
                          << node_executor_name_ << std::endl << std::flush;
  launchExecutionThread();
}
#else
TensorRuntime::TensorRuntime(const ParamConf & parameters,
                             const std::string & graph_executor_name,
                             const std::string & node_executor_name):
 parameters_(parameters),
 graph_executor_name_(graph_executor_name), node_executor_name_(node_executor_name), current_dag_(nullptr),
 logging_(0), backend_(CompBackend::None), executing_(false), scope_set_(false), alive_(false)
{
#ifdef DEBUG
  const bool debugging = true;
#else
  const bool debugging = false;
#endif
  num_processes_ = 1; process_rank_ = 0; global_process_rank_ = 0;
  graph_executor_ = exatn::getService<TensorGraphExecutor>(graph_executor_name_);
  if(debugging) std::cout << "#DEBUG(exatn::runtime::TensorRuntime)[MAIN_THREAD]: DAG executor set to "
                          << graph_executor_name_ << " + " << node_executor_name_ << std::endl << std::flush;
  launchExecutionThread();
}
#endif


TensorRuntime::~TensorRuntime()
{
  if(alive_.load()){
    alive_.store(false); //signal for the execution thread to finish
    //std::cout << "#DEBUG(exatn::runtime::TensorRuntime)[MAIN_THREAD]: Waiting Execution Thread ... " << std::flush;
    exec_thread_.join(); //wait until the execution thread has finished
    //std::cout << "Joined" << std::endl << std::flush;
  }
}


void TensorRuntime::launchExecutionThread()
{
  if(!(alive_.load())){
    alive_.store(true);
    //std::cout << "#DEBUG(exatn::runtime::TensorRuntime)[MAIN_THREAD]: Launching Execution Thread ... " << std::flush;
    exec_thread_ = std::thread(&TensorRuntime::executionThreadWorkflow,this);
    //std::cout << "Done" << std::endl << std::flush;
  }
  return; //only the main thread returns to the client
}


void TensorRuntime::executionThreadWorkflow()
{
  graph_executor_->resetNodeExecutor(exatn::getService<TensorNodeExecutor>(node_executor_name_),
                                     parameters_,num_processes_,process_rank_,global_process_rank_);
  //std::cout << "#DEBUG(exatn::runtime::TensorRuntime)[EXEC_THREAD]: DAG node executor set to "
            //<< node_executor_name_ << std::endl << std::flush;
  while(alive_.load()){ //alive_ is set by the main thread
    while(executing_.load()){ //executing_ is set to TRUE by the main thread when new operations and syncs are submitted
      graph_executor_->execute(*current_dag_);
      processTensorDataRequests(); //process all outstanding client requests for tensor data (synchronous)
      if(current_dag_->hasUnexecutedNodes()){
        executing_.store(true); //reaffirm that DAG is still executing
      }else{
        graph_executor_->execute(tensor_network_queue_);
        if(!(current_dag_->hasUnexecutedNodes())) executing_.store(false); //executing_ is set to FALSE by the execution thread
      }
    }
    processTensorDataRequests(); //process all outstanding client requests for tensor data (synchronous)
  }
  graph_executor_->resetNodeExecutor(std::shared_ptr<TensorNodeExecutor>(nullptr),
                                     parameters_,num_processes_,process_rank_,global_process_rank_);
  //std::cout << "#DEBUG(exatn::runtime::TensorRuntime)[EXEC_THREAD]: DAG node executor reset. End of life."
            //<< std::endl << std::flush;
  return; //end of execution thread life
}


void TensorRuntime::switchCompBackend(CompBackend requested_backend)
{
  bool synced = true;
  if(backend_ != CompBackend::None){
    if(backend_ != requested_backend) synced = sync(backend_,true);
  }
  assert(synced);
  backend_ = requested_backend;
  return;
}


void TensorRuntime::processTensorDataRequests()
{
  lockDataReqQ();
  for(auto & req: data_req_queue_){
    req.slice_promise_.set_value(graph_executor_->getLocalTensor(*(req.tensor_),req.slice_specs_));
  }
  data_req_queue_.clear();
  unlockDataReqQ();
  return;
}


void TensorRuntime::resetLoggingLevel(int level)
{
  while(!graph_executor_);
  graph_executor_->resetLoggingLevel(level);
  logging_ = level;
  return;
}


void TensorRuntime::resetSerialization(bool serialize, bool validation_trace)
{
  while(!graph_executor_);
  return graph_executor_->resetSerialization(serialize,validation_trace);
}


void TensorRuntime::activateDryRun(bool dry_run)
{
  while(!graph_executor_);
  return graph_executor_->activateDryRun(dry_run);
}


void TensorRuntime::activateFastMath()
{
  while(!graph_executor_);
  return graph_executor_->activateFastMath();
}


std::size_t TensorRuntime::getMemoryBufferSize() const
{
  while(!graph_executor_);
  return graph_executor_->getMemoryBufferSize();
}


std::size_t TensorRuntime::getMemoryUsage(std::size_t * free_mem) const
{
  while(!graph_executor_);
  return graph_executor_->getMemoryUsage(free_mem);
}


double TensorRuntime::getTotalFlopCount() const
{
  while(!graph_executor_);
  return graph_executor_->getTotalFlopCount();
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
  current_dag_ = (new_dag.first)->second; //storing a shared pointer to the DAG
  current_scope_ = scope_name; // change the name of the current scope
  scope_set_.store(true);
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
  current_dag_ = dags_[scope_name]; //storing a shared pointer to the DAG
  current_scope_ = scope_name; // change the name of the current scope
  scope_set_.store(true);
  executing_.store(true); //will trigger DAG execution by the execution thread
  return;
}


void TensorRuntime::closeScope() {
  if(currentScopeIsSet()){
    sync();
    while(executing_.load()){}; //wait until the execution thread has completed execution of the current DAG
    const std::string scope_name = current_scope_;
    scope_set_.store(false);
    current_scope_ = "";
    current_dag_.reset();
    auto num_deleted = dags_.erase(scope_name);
    assert(num_deleted == 1);
  }
  return;
}


VertexIdType TensorRuntime::submit(std::shared_ptr<TensorOperation> op) {
  assert(currentScopeIsSet());
  switchCompBackend(CompBackend::Default);
  auto node_id = current_dag_->addOperation(op);
  op->setId(node_id);
  //current_dag_->printIt(); //debug
  executing_.store(true); //signal to the execution thread to execute the DAG
  return node_id;
}


bool TensorRuntime::sync(TensorOperation & op, bool wait) {
  assert(currentScopeIsSet());
  executing_.store(true); //reactivate the execution thread to execute the DAG in case it was not active
  auto opid = op.getId();
  bool completed = current_dag_->nodeExecuted(opid);
  while(wait && (!completed)){
    executing_.store(true); //reactivate the execution thread to execute the DAG in case it was not active
    completed = current_dag_->nodeExecuted(opid);
  }
  return completed;
}


bool TensorRuntime::sync(const Tensor & tensor, bool wait) {
  //if(wait) std::cout << "#DEBUG(TensorRuntime::sync)[MAIN_THREAD]: Syncing on tensor " << tensor.getName() << " ... "; //debug
  assert(currentScopeIsSet());
  executing_.store(true); //reactivate the execution thread to execute the DAG in case it was not active
  bool completed = (current_dag_->getTensorUpdateCount(tensor) == 0);
  while(wait && (!completed)){
    executing_.store(true); //reactivate the execution thread to execute the DAG in case it was not active
    completed = (current_dag_->getTensorUpdateCount(tensor) == 0);
  }
  //if(wait) std::cout << "Synced" << std::endl; //debug
  return completed;
}


bool TensorRuntime::sync(CompBackend backend, bool wait) {
  bool synced = true;
  switch(backend){
  case CompBackend::None:
    break;
  case CompBackend::Default:
    synced = syncTensOps(wait);
    break;
#ifdef CUQUANTUM
  case CompBackend::Cuquantum:
    synced = syncNetworks(wait);
    break;
#endif
  }
  return synced;
}


bool TensorRuntime::sync(bool wait) {
  bool synced = sync(backend_,wait);
  if(synced && backend_ != CompBackend::Default) synced = sync(CompBackend::Default,wait);
#ifdef CUQUANTUM
  if(synced && backend_ != CompBackend::Cuquantum) synced = sync(CompBackend::Cuquantum,wait);
#endif
  return synced;
}


bool TensorRuntime::syncTensOps(bool wait) {
  //if(wait) std::cout << "#DEBUG(TensorRuntime::syncTensOps)[MAIN_THREAD]: Syncing default backend ... "; //debug
  assert(currentScopeIsSet());
  if(current_dag_->hasUnexecutedNodes()) executing_.store(true);
  bool still_working = executing_.load();
  while(wait && still_working){
    if(current_dag_->hasUnexecutedNodes()) executing_.store(true);
    still_working = executing_.load();
  }
  if(wait && (!still_working)){
    if(current_dag_->getNumNodes() > MAX_RUNTIME_DAG_SIZE){
      //std::cout << "Clearing DAG ... "; //debug
      current_dag_->clear();
      //std::cout << "Done; "; //debug
    }
  }
  //if(wait) std::cout << "Synced\n" << std::flush; //debug
  return !still_working;
}


#ifdef CUQUANTUM
TensorOpExecHandle TensorRuntime::submit(std::shared_ptr<numerics::TensorNetwork> network,
                                         const MPICommProxy & communicator,
                                         unsigned int num_processes, unsigned int process_rank)
{
  switchCompBackend(CompBackend::Cuquantum);
  const auto exec_handle = tensor_network_queue_.append(network,communicator,num_processes,process_rank);
  executing_.store(true); //signal to the execution thread to execute the queue
  return exec_handle;
}


bool TensorRuntime::syncNetwork(const TensorOpExecHandle exec_handle, bool wait)
{
  assert(exec_handle != 0);
  executing_.store(true); //reactivate the execution thread in case it was not active
  bool synced = false;
  while(!synced){
    const auto exec_stat = tensor_network_queue_.checkExecStatus(exec_handle);
    synced = (exec_stat == TensorNetworkQueue::ExecStat::None ||
              exec_stat == TensorNetworkQueue::ExecStat::Completed);
    if(!wait) break;
  };
  return synced;
}


bool TensorRuntime::syncNetworks(bool wait)
{
  //if(wait) std::cout << "#DEBUG(TensorRuntime::syncNetworks)[MAIN_THREAD]: Syncing cuQuantum backend ... "; //debug
  executing_.store(true); //reactivate the execution thread in case it was not active
  bool synced = false;
  while(!synced){
    synced = tensor_network_queue_.isEmpty();
    if(!wait) break;
  }
  //if(wait) std::cout << "Synced\n" << std::flush; //debug
  return synced;
}
#endif


std::future<std::shared_ptr<talsh::Tensor>> TensorRuntime::getLocalTensor(std::shared_ptr<Tensor> tensor,
                                          const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec)
{
  // Complete all submitted update operations on the tensor:
  auto synced = sync(*tensor,true); assert(synced);
  // Create promise-future pair:
  std::promise<std::shared_ptr<talsh::Tensor>> promised_slice;
  auto future_slice = promised_slice.get_future();
  // Schedule data request:
  lockDataReqQ();
  data_req_queue_.emplace_back(std::move(promised_slice),slice_spec,tensor);
  unlockDataReqQ();
  return future_slice;
}

} // namespace runtime
} // namespace exatn
