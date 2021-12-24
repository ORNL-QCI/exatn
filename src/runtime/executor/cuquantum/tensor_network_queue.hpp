/** ExaTN: Tensor Runtime: Tensor network executor: Execution queue
REVISION: 2021/12/24

Copyright (C) 2018-2021 Dmitry Lyakh
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 - ExaTN graph executor may accept whole tensor networks for execution
   via the optional cuQuantum backend in which case the graph executor
   will delegate execution of whole tensor networks to CuQuantumExecutor.

**/

#ifndef EXATN_RUNTIME_TENSOR_NETWORK_QUEUE_HPP_
#define EXATN_RUNTIME_TENSOR_NETWORK_QUEUE_HPP_

#include "tensor_network.hpp"
#include "tensor_operation.hpp"

#include <list>
#include <memory>
#include <atomic>
#include <mutex>

#include "errors.hpp"

namespace exatn {
namespace runtime {

class TensorNetworkQueue {

public:

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

 bool isEmpty() {
  lock();
  bool empt = networks_.empty();
  unlock();
  return empt;
 }

 std::size_t getSize() {
  lock();
  const std::size_t current_size = networks_.size();
  unlock();
  return current_size;
 }

 TensorOpExecHandle append(std::shared_ptr<numerics::TensorNetwork> network) {
  lock();
  const TensorOpExecHandle tn_hash = getTensorNetworkHash(network);
  networks_.emplace_back(std::make_pair(network,tn_hash));
  unlock();
  return tn_hash;
 }

 void remove() {
  lock();
  assert(current_network_ != networks_.end());
  current_network_ = networks_.erase(current_network_);
  unlock();
  return;
 }

 ConstTensorNetworkQueueIterator getCurrent() {
  return current_network_;
 }

 void reset() {
  lock();
  current_network_ = networks_.begin();
  unlock();
  return;
 }

 bool isOver() {
  lock();
  bool over = (current_network_ == networks_.end());
  unlock();
  return over;
 }

 bool next() {
  lock();
  assert(current_network_ != networks_.end());
  ++current_network_;
  bool not_over = (current_network_ != networks_.end());
  unlock();
  return not_over;
 }

 inline void lock(){queue_lock_.lock();}
 inline void unlock(){queue_lock_.unlock();}

protected:

 /** Queue of tensor networks to be executed **/
 std::list<std::pair<std::shared_ptr<numerics::TensorNetwork>,
                     TensorOpExecHandle>> networks_;
 TensorNetworkQueueIterator current_network_;
 std::mutex queue_lock_;
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_TENSOR_NETWORK_QUEUE_HPP_
