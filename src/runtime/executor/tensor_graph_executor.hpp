/** ExaTN:: Tensor Runtime: Tensor graph executor
REVISION: 2021/02/20

Copyright (C) 2018-2021 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 (a) Tensor graph executor traverses the tensor graph (DAG) and
     executes all its nodes while respecting node dependencies.
     Each DAG node is executed by a concrete tensor node executor
     (tensor operation stored in the DAG node accepts a polymorphic
     tensor node executor which then executes that tensor operation).
     The execution of each DAG node is generally asynchronous.
**/

#ifndef EXATN_RUNTIME_TENSOR_GRAPH_EXECUTOR_HPP_
#define EXATN_RUNTIME_TENSOR_GRAPH_EXECUTOR_HPP_

#include "Identifiable.hpp"

#include "tensor_graph.hpp"
#include "tensor_node_executor.hpp"
#include "tensor_operation.hpp"

#include "param_conf.hpp"

#include "timers.hpp"

#include <memory>
#include <atomic>

#include <iostream>
#include <fstream>
#include <iomanip>

#include "errors.hpp"

namespace exatn {
namespace runtime {

class TensorGraphExecutor : public Identifiable, public Cloneable<TensorGraphExecutor> {

public:

  TensorGraphExecutor():
   node_executor_(nullptr), num_ops_issued_(0), process_rank_(-1), global_process_rank_(-1),
   logging_(0), stopping_(false), active_(false), time_start_(exatn::Timer::timeInSecHR())
  {
  }

  TensorGraphExecutor(const TensorGraphExecutor &) = delete;
  TensorGraphExecutor & operator=(const TensorGraphExecutor &) = delete;
  TensorGraphExecutor(TensorGraphExecutor &&) noexcept = delete;
  TensorGraphExecutor & operator=(TensorGraphExecutor &&) noexcept = delete;

  virtual ~TensorGraphExecutor(){
    resetLoggingLevel();
  }

  /** Sets/resets the DAG node executor (tensor operation executor). **/
  void resetNodeExecutor(std::shared_ptr<TensorNodeExecutor> node_executor,
                         const ParamConf & parameters,
                         unsigned int process_rank,
                         unsigned int global_process_rank) {
    process_rank_.store(process_rank);
    global_process_rank_.store(global_process_rank);
    node_executor_ = node_executor;
    if(node_executor_){
      if(logging_.load() != 0){
        logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
                 << "](TensorGraphExecutor)[EXEC_THREAD]: Initializing the node executor ... "; //debug
      }
      node_executor_->initialize(parameters);
      if(logging_.load() != 0){
        logfile_ << "Successfully initialized [" << std::fixed << std::setprecision(6)
                 << exatn::Timer::timeInSecHR(getTimeStampStart()) << "]" << std::endl; //debug
        logfile_.flush();
      }
    }
    return;
  }

  /** Resets the logging level (0:none). **/
  void resetLoggingLevel(int level = 0) {
    if(logging_.load() == 0){
      while(level != 0 && global_process_rank_.load() < 0);
      if(level != 0) logfile_.open("exatn_exec_thread."+std::to_string(global_process_rank_.load())+".log", std::ios::out | std::ios::trunc);
    }else{
      if(level == 0) logfile_.close();
    }
    logging_.store(level);
    return;
  }

  /** Activates mixed-precision fast math on all devices (if available). **/
  void activateFastMath() {
    while(!node_executor_);
    return node_executor_->activateFastMath();
  }

  /** Returns the Host memory buffer size in bytes provided by the node executor. **/
  std::size_t getMemoryBufferSize() const {
    while(!node_executor_);
    return node_executor_->getMemoryBufferSize();
  }

  /** Returns the current value of the total Flop count executed by the node executor. **/
  double getTotalFlopCount() const {
    while(!node_executor_);
    return node_executor_->getTotalFlopCount();
  }

  /** Traverses the DAG and executes all its nodes (operations).
      [THREAD: This function is executed by the execution thread] **/
  virtual void execute(TensorGraph & dag) = 0;

  /** Regulates the tensor prefetch depth (0 turns prefetch off). **/
  virtual void setPrefetchDepth(unsigned int depth) = 0;

  /** Factory method **/
  virtual std::shared_ptr<TensorGraphExecutor> clone() = 0;

  /** Returns a local copy of a given tensor slice. **/
  std::shared_ptr<talsh::Tensor> getLocalTensor(const numerics::Tensor & tensor,
                 const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec) {
    assert(node_executor_);
    return node_executor_->getLocalTensor(tensor,slice_spec);
  }

  /** Signals to stop execution of the DAG until later resume
      and waits until the execution has actually stopped.
      [THREAD: This function is executed by the main thread] **/
  void stopExecution() {
    stopping_.store(true);   //this signal will be picked by the execution thread
    while(active_.load()){}; //once the DAG execution is stopped the execution thread will set active_ to FALSE
    return;
  }

  inline double getTimeStampStart() const {return time_start_;}

  inline std::size_t incrementOpCounter() {return ++num_ops_issued_;}

  inline std::size_t getOpCounter() const {return num_ops_issued_.load();}

protected:

  std::shared_ptr<TensorNodeExecutor> node_executor_; //intr-node tensor operation executor
  std::atomic<std::size_t> num_ops_issued_; //total number of issued tensor operations
  std::atomic<int> process_rank_; //current process rank
  std::atomic<int> global_process_rank_; //current global process rank (in MPI_COMM_WORLD)
  std::atomic<int> logging_;      //logging level (0:none)
  std::atomic<bool> stopping_;    //signal to pause the execution thread
  std::atomic<bool> active_;      //TRUE while the execution thread is executing DAG operations
  const double time_start_;       //start time stamp
  std::ofstream logfile_;         //logging file stream (output)
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_TENSOR_GRAPH_EXECUTOR_HPP_
