/** ExaTN: Tensor Runtime: Tensor network executor: Execution queue
REVISION: 2022/01/07

Copyright (C) 2018-2022 Dmitry Lyakh
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 - ExaTN graph executor may accept whole tensor networks for execution
   via the optional cuQuantum backend in which case the graph executor
   will delegate execution of whole tensor networks to CuQuantumExecutor.

**/

#ifndef EXATN_RUNTIME_TENSOR_NETWORK_QUEUE_HPP_
#define EXATN_RUNTIME_TENSOR_NETWORK_QUEUE_HPP_

#include "tensor_network.hpp"
#include "tensor_operation.hpp"

#include <unordered_map>
#include <list>
#include <memory>
#include <atomic>
#include <mutex>

#include "errors.hpp"

namespace exatn {
namespace runtime {

class TensorNetworkQueue {

public:

 //Tensor network execution status:
 enum class ExecStat {
  None,      //no execution status
  Idle,      //submitted but execution has not yet started
  Loading,   //started loading data
  Planning,  //preparation for execution has started (planning)
  Executing, //actual execution (numerical computation) has started
  Completed  //execution completed
 };

 //Tensor network entry:
 struct NetEntry {
  ExecStat exec_status = ExecStat::None; //execution status
  unsigned int num_procs = 0;            //number of executing processes
  unsigned int proc_id = 0;              //id of the current executing process
  MPICommProxy comm;                     //MPI communicator proxy
 };

 using TensorNetworkQueueIterator =
  std::list<std::pair<std::shared_ptr<numerics::TensorNetwork>,TensorOpExecHandle>>::iterator;

 using ConstTensorNetworkQueueIterator =
  std::list<std::pair<std::shared_ptr<numerics::TensorNetwork>,TensorOpExecHandle>>::const_iterator;

 TensorNetworkQueue(): current_network_(networks_.end()) {
 }

 TensorNetworkQueue(const TensorNetworkQueue &) = delete;
 TensorNetworkQueue & operator=(const TensorNetworkQueue &) = delete;
 TensorNetworkQueue(TensorNetworkQueue &&) noexcept = delete;
 TensorNetworkQueue & operator=(TensorNetworkQueue &&) noexcept = delete;
 ~TensorNetworkQueue() = default;

 TensorNetworkQueueIterator begin() {return networks_.begin();}
 TensorNetworkQueueIterator end() {return networks_.end();}
 ConstTensorNetworkQueueIterator cbegin() {return networks_.cbegin();}
 ConstTensorNetworkQueueIterator cend() {return networks_.cend();}

 /** Returns TRUE is the tensor network queue is empty, FALSE otherwise. **/
 bool isEmpty() {
  lock();
  bool empt = networks_.empty();
  unlock();
  return empt;
 }

 /** Returns the current size of the tensor network queue. **/
 std::size_t getSize() {
  lock();
  const std::size_t current_size = networks_.size();
  unlock();
  return current_size;
 }

 /** Appends a new tensor network to the queue (no repeats allowed).
     Upon success, returns a positive execution handle, zero otherwise. **/
 TensorOpExecHandle append(std::shared_ptr<numerics::TensorNetwork> network,
                           const MPICommProxy & communicator,
                           unsigned int num_processes,
                           unsigned int process_rank) {
  lock();
  TensorOpExecHandle tn_hash = getTensorNetworkHash(network);
  auto res = tn_exec_stat_.emplace(std::make_pair(tn_hash,
              NetEntry{ExecStat::Idle,num_processes,process_rank,communicator}));
  if(res.second){
   networks_.emplace_back(std::make_pair(network,tn_hash));
  }else{
   tn_hash = 0;
  }
  unlock();
  return tn_hash;
 }

 /** Removes the tensor network currently pointed to from the queue.
     The tensor network execution status must be marked Completed. **/
 void remove() {
  lock();
  assert(current_network_ != networks_.end());
  auto iter = tn_exec_stat_.find(current_network_->second);
  if(iter != tn_exec_stat_.end()){
   if(iter->second.exec_status == ExecStat::Completed){
    tn_exec_stat_.erase(iter);
   }else{
    std::cout << "#ERROR(exatn::runtime::TensorNetworkQueue): Attempt to delete an unfinished tensor network!\n";
    assert(false);
   }
  }
  current_network_ = networks_.erase(current_network_);
  unlock();
  return;
 }

 /** Returns the execution status associated with
     the given tensor network execution handle. **/
 ExecStat checkExecStatus(const TensorOpExecHandle exec_handle) {
  auto exec_stat = ExecStat::None;
  lock();
  auto iter = tn_exec_stat_.find(exec_handle);
  if(iter != tn_exec_stat_.cend()) exec_stat = iter->second.exec_status;
  unlock();
  return exec_stat;
 }

 /** Updates the execution status associated with
     the given tensor network execution handle.
     Returns the previous execution status. **/
 ExecStat updateExecStatus(const TensorOpExecHandle exec_handle,
                           ExecStat new_exec_stat) {
  auto exec_stat = ExecStat::None;
  lock();
  auto iter = tn_exec_stat_.find(exec_handle);
  if(iter != tn_exec_stat_.end()){
   exec_stat = iter->second.exec_status;
   iter->second.exec_status = new_exec_stat;
  }
  unlock();
  return exec_stat;
 }

 /** Returns the parallel execution configuration associated
     with the given tensor network execution handle. **/
 std::pair<int,int> getExecConfiguration(const TensorOpExecHandle exec_handle,
                                         MPICommProxy * communicator = nullptr) {
  std::pair<int,int> exec_conf{0,-1};
  lock();
  auto iter = tn_exec_stat_.find(exec_handle);
  if(iter != tn_exec_stat_.cend()){
   exec_conf = std::make_pair(iter->second.num_procs,iter->second.proc_id);
   if(communicator != nullptr) *communicator = iter->second.comm;
  }
  unlock();
  return exec_conf;
 }

 /** Returns the constant iterator to the current tensor network. **/
 ConstTensorNetworkQueueIterator getCurrent() {
  return current_network_;
 }

 /** Resets the current iterator to the beginning of the queue. **/
 void reset() {
  lock();
  current_network_ = networks_.begin();
  unlock();
  return;
 }

 /** Returns TRUE if the current iterator is positioned
     after the end of the queue, FALSE otherwise. **/
 bool isOver() {
  lock();
  bool over = (current_network_ == networks_.end());
  unlock();
  return over;
 }

 /** Moves the current iterator to the next element of the queue.
     If moved past the end, return FALSE, otherwise TRUE.
     The current iterator must be valid on entrance. **/
 bool next() {
  lock();
  assert(current_network_ != networks_.end());
  ++current_network_;
  bool not_over = (current_network_ != networks_.end());
  unlock();
  return not_over;
 }

 /** Returns the distance from the current iterator
     to the beginning of the queue. **/
 auto getCurrentPos() {
  return std::distance(networks_.begin(),current_network_);
 }

 /** Locks. **/
 inline void lock(){queue_lock_.lock();}
 inline void unlock(){queue_lock_.unlock();}

protected:

 /** Tensor network execution status **/
 std::unordered_map<TensorOpExecHandle,NetEntry> tn_exec_stat_;
 /** Queue of tensor networks to be executed **/
 std::list<std::pair<std::shared_ptr<numerics::TensorNetwork>,
                     TensorOpExecHandle>> networks_;
 /** Tensor network iterator **/
 TensorNetworkQueueIterator current_network_;
 std::mutex queue_lock_;
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_TENSOR_NETWORK_QUEUE_HPP_
