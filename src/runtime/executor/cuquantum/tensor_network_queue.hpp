/** ExaTN: Tensor Runtime: Tensor network executor: Execution queue
REVISION: 2021/12/21

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

 TensorNetworkQueue() = default;
 TensorNetworkQueue(const TensorNetworkQueue &) = delete;
 TensorNetworkQueue & operator=(const TensorNetworkQueue &) = delete;
 TensorNetworkQueue(TensorNetworkQueue &&) noexcept = delete;
 TensorNetworkQueue & operator=(TensorNetworkQueue &&) noexcept = delete;
 ~TensorNetworkQueue() = default;

 inline void lock(){queue_lock_.lock();}
 inline void unlock(){queue_lock_.unlock();}

protected:

 std::list<std::pair<std::shared_ptr<numerics::TensorNetwork>,
                     TensorOpExecHandle>> networks_;
 std::mutex queue_lock_;
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_TENSOR_NETWORK_QUEUE_HPP_
