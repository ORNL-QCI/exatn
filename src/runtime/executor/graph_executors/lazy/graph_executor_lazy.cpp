/** ExaTN:: Tensor Runtime: Tensor graph executor: Lazy
REVISION: 2020/06/16

Copyright (C) 2018-2020 Dmitry Lyakh
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle)
**/

#include "graph_executor_lazy.hpp"

#include "talshxx.hpp"

#include <iostream>
#include <iomanip>

#include <cassert>

namespace exatn {
namespace runtime {

void LazyGraphExecutor::execute(TensorGraph & dag) {

  auto num_nodes = dag.getNumNodes();
  auto front = dag.getFrontNode();
  auto current = front;

  auto move_to_next_node = [this,&dag,&num_nodes,&front,&current] () {
    bool moved = false;
    num_nodes = dag.getNumNodes();
    if(++current < num_nodes){
      front = dag.getFrontNode();
      if(current < (front + this->getPipelineDepth())){
        moved = true;
      }else{
        current = front;
        moved = true;
      }
    }
    return moved;
  };

  auto inspect_node_dependencies = [&dag,&current](){
    bool ready_for_execution = dag.nodeDependenciesResolved(current);
    if(ready_for_execution) dag.registerDependencyFreeNode(current);
    return ready_for_execution;
  };

  auto activate_node_prefetch = [this](TensorOpNode & dag_node){
    bool prefetch_activated = this->node_executor_->prefetch(*(dag_node.getOperation()));
    return prefetch_activated;
  };

  auto issue_ready_node = [this,&dag](){
    VertexIdType node;
    bool issued = dag.extractDependencyFreeNode(&node);
    if(issued){
      dag.setNodeExecuting(node);
      auto & dag_node = dag.getNodeProperties(node);
      auto op = dag_node.getOperation();
      op->recordStartTime();
      TensorOpExecHandle exec_handle;
      auto error_code = op->accept(*(this->node_executor_),&exec_handle);
      if(error_code == 0){
        dag.registerExecutingNode(node,exec_handle);
      }else{
        auto discarded = this->node_executor_->discard(exec_handle);
        dag.setNodeIdle(node);
        issued = false;
        if(error_code != TRY_LATER && error_code != DEVICE_UNABLE){
          std::cout << "#ERROR(exatn::TensorRuntime::GraphExecutorLazy): Failed to submit tensor operation "
           << node << " with execution handle " << exec_handle << ": Error " << error_code << std::endl << std::flush;
        }
      }
    }
    return issued;
  };

  auto test_nodes_for_completion = [this,&dag,&front](){
    auto executing_nodes = dag.executingNodesBegin();
    while(executing_nodes != dag.executingNodesEnd()){
      int error_code;
      auto exec_handle = executing_nodes->second;
      auto synced = this->node_executor_->sync(exec_handle,&error_code,false);
      if(synced){
        VertexIdType node;
        executing_nodes = dag.extractExecutingNode(executing_nodes,&node);
        auto & dag_node = dag.getNodeProperties(node);
        auto op = dag_node.getOperation();
        op->recordFinishTime();
        if(error_code == 0){
          dag.setNodeExecuted(node,error_code);
          dag.progressFrontNode(node);
          front = dag.getFrontNode();
        }else{
          auto discarded = this->node_executor_->discard(exec_handle);
          dag.setNodeExecuted(node,error_code);
          std::cout << "#ERROR(exatn::TensorRuntime::GraphExecutorLazy): Failed to synchronize tensor operation "
           << node << " with execution handle " << exec_handle << ": Error " << error_code << std::endl << std::flush;
        }
      }else{
        ++executing_nodes;
      }
    }
    return;
  };


  while(front < num_nodes){
    TensorOpExecHandle exec_handle;
    auto & dag_node = dag.getNodeProperties(current);
    if(!(dag_node.isExecuted())){
      dag.setNodeExecuting(current);
      auto op = dag_node.getOperation();
      op->recordStartTime();
      auto error_code = op->accept(*node_executor_,&exec_handle);
      if(error_code == 0){
        auto synced = node_executor_->sync(exec_handle,&error_code,true);
        op->recordFinishTime();
        if(synced && error_code == 0){
          dag.setNodeExecuted(current);
          dag.progressFrontNode(current);
          ++current;
        }else{ //failed to synchronize the submitted tensor operation
          node_executor_->discard(exec_handle);
          if(error_code != 0) dag.setNodeExecuted(current,error_code);
          std::cout << "#ERROR(exatn::TensorRuntime::GraphExecutorLazy): Failed to synchronize tensor operation: Error "
                    << error_code << std::endl << std::flush;
          assert(false);
        }
      }else{ //failed to submit the tensor operation
        node_executor_->discard(exec_handle);
        dag.setNodeIdle(current);
        if(error_code != TRY_LATER && error_code != DEVICE_UNABLE){
          std::cout << "#ERROR(exatn::TensorRuntime::GraphExecutorLazy): Failed to submit tensor operation: Error "
                    << error_code << std::endl << std::flush;
          assert(false);
        }
      }
    }else{
      ++current;
    }
    num_nodes = dag.getNumNodes();
  }


  return;
}

#if 0
if(logging_.load() != 0){
        logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
                 << "](LazyGraphExecutor)[EXEC_THREAD]: Submitting tensor operation "
                 << current << ": Opcode = " << static_cast<int>(op->getOpcode()); //debug
        if(logging_.load() > 1){
          logfile_ << ": Details:" << std::endl;
          op->printItFile(logfile_);
        }
      }

if(logging_.load() != 0){
        logfile_ << ": Status = " << error_code << ": "; //debug
      }

if(logging_.load() != 0){
          logfile_ << "Syncing ... "; //debug
        }

if(logging_.load() != 0){
            logfile_ << "Success [" << std::fixed << std::setprecision(6)
                     << exatn::Timer::timeInSecHR(getTimeStampStart()) << "]" << std::endl; //debug
            logfile_.flush();
          }

if(logging_.load() != 0){
            logfile_ << "Failed to synchronize tensor operation: Error " << error_code << std::endl;
            logfile_.flush();
          }

if(logging_.load() != 0){
          logfile_ << "Will retry again" << std::endl; //debug
          logfile_.flush();
        }
#endif

} //namespace runtime
} //namespace exatn
