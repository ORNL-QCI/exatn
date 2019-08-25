/** ExaTN:: Tensor Runtime: Tensor graph execution state
REVISION: 2019/08/04

Copyright (C) 2018-2019 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)
**/

#include "tensor_exec_state.hpp"

#include <iostream>
#include <assert.h>

namespace exatn {
namespace runtime {

const std::vector<VertexIdType> * TensorExecState::getTensorEpochNodes(const Tensor & tensor, int * epoch)
{
  *epoch = 0;
  auto tens_hash = tensor.getTensorHash();
  auto iter = tensor_info_.find(tens_hash);
  if(iter == tensor_info_.end()) return nullptr;
  *epoch = (iter->second).rw_epoch;
  return &((iter->second).rw_epoch_nodes);
}

int TensorExecState::registerTensorRead(const Tensor & tensor, VertexIdType node_id)
{
  auto tens_hash = tensor.getTensorHash();
  auto iter = tensor_info_.find(tens_hash);
  if(iter == tensor_info_.end()){
    auto pos = tensor_info_.emplace(std::make_pair(tens_hash,TensorExecInfo()));
    iter = pos.first;
  }
  auto & tens_info = iter->second;
  if(tens_info.rw_epoch < 0){ //write epoch
    tens_info.rw_epoch_nodes.clear();
    tens_info.rw_epoch = 0;
  }
  tens_info.rw_epoch_nodes.emplace_back(node_id);
  return ++(tens_info.rw_epoch);
}

int TensorExecState::registerTensorWrite(const Tensor & tensor, VertexIdType node_id)
{
  auto tens_hash = tensor.getTensorHash();
  auto iter = tensor_info_.find(tens_hash);
  if(iter == tensor_info_.end()){
    auto pos = tensor_info_.emplace(std::make_pair(tens_hash,TensorExecInfo()));
    iter = pos.first;
  }
  auto & tens_info = iter->second;
  if(tens_info.rw_epoch != 0){ //either read or write epoch
    tens_info.rw_epoch_nodes.clear();
    tens_info.rw_epoch = 0;
  }
  tens_info.rw_epoch_nodes.emplace_back(node_id);
  ++(tens_info.update_count);
  return --(tens_info.rw_epoch); //-1
}

std::size_t TensorExecState::registerWriteCompletion(const Tensor & tensor)
{
  auto tens_hash = tensor.getTensorHash();
  auto iter = tensor_info_.find(tens_hash);
  assert(iter != tensor_info_.end());
  return --((iter->second).update_count);
}

std::size_t TensorExecState::getTensorUpdateCount(const Tensor & tensor)
{
  auto tens_hash = tensor.getTensorHash();
  auto iter = tensor_info_.find(tens_hash);
  if(iter == tensor_info_.end()) return 0;
  return (iter->second).update_count;
}

void TensorExecState::registerDependencyFreeNode(VertexIdType node_id)
{
  nodes_ready_.emplace_back(node_id);
  return;
}

bool TensorExecState::extractDependencyFreeNode(VertexIdType * node_id)
{
  bool empty = nodes_ready_.empty();
  if(!empty){
    *node_id = nodes_ready_.front();
    nodes_ready_.pop_front();
  }
  return !empty;
}

void TensorExecState::registerExecutingNode(VertexIdType node_id)
{
  nodes_executing_.emplace_back(node_id);
  return;
}

bool TensorExecState::extractExecutingNode(VertexIdType * node_id)
{
  bool empty = nodes_executing_.empty();
  if(!empty){
    *node_id = nodes_executing_.front();
    nodes_executing_.pop_front();
  }
  return !empty;
}

} // namespace runtime
} // namespace exatn
