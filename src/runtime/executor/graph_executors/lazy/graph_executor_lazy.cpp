/** ExaTN:: Tensor Runtime: Tensor graph executor: Lazy
REVISION: 2020/06/17

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

  struct Progress {
    VertexIdType num_nodes;
    VertexIdType front;
    VertexIdType current;
  };

  Progress progress{dag.getNumNodes(),dag.getFrontNode(),0};
  progress.current = progress.front;

  auto move_to_next_node = [this,&dag,&progress] () {
    bool moved = false;
    progress.num_nodes = dag.getNumNodes();
    if(++(progress.current) < progress.num_nodes){
      progress.front = dag.getFrontNode();
      if(progress.current < (progress.front + this->getPipelineDepth())){
        moved = true;
      }else{ //the current node must stay within the max pipeline depth from the front node
        progress.current = progress.front;
        moved = true;
      }
    }
    return moved;
  };

  auto inspect_node_dependencies = [this,&dag,&progress] () {
    auto & dag_node = dag.getNodeProperties(progress.current);
    bool ready_for_execution = dag_node.isIdle();
    if(ready_for_execution){ //node is idle
      ready_for_execution = ready_for_execution && dag.nodeDependenciesResolved(progress.current);
      if(ready_for_execution){ //all node dependencies resolved (or none)
        dag.registerDependencyFreeNode(progress.current);
      }else{ //node still has unresolved dependencies, try prefetching
        if(progress.current < (progress.front + this->getPrefetchDepth())){
          auto prefetching = this->node_executor_->prefetch(*(dag_node.getOperation()));
        }
      }
    }
    return ready_for_execution;
  };

  auto issue_ready_node = [this,&dag,&progress] () {
    VertexIdType node;
    bool issued = dag.extractDependencyFreeNode(&node);
    if(issued){
      dag.setNodeExecuting(node);
      auto & dag_node = dag.getNodeProperties(node);
      auto op = dag_node.getOperation();
      op->recordStartTime();
      TensorOpExecHandle exec_handle;
      auto error_code = op->accept(*(this->node_executor_),&exec_handle);
      if(error_code == 0){ //tensor operation submitted for execution successfully
        auto synced = this->node_executor_->sync(exec_handle,&error_code,false);
        if(synced){ //tensor operation has completed immediately
          op->recordFinishTime();
          dag.setNodeExecuted(node,error_code);
          if(error_code == 0){
            dag.progressFrontNode(node);
            progress.front = dag.getFrontNode();
          }else{
            std::cout << "#ERROR(exatn::TensorRuntime::GraphExecutorLazy): Immediate completion error for tensor operation "
             << node << " with execution handle " << exec_handle << ": Error " << error_code << std::endl << std::flush;
            assert(false); //`Do I need to handle this case gracefully?
          }
        }else{ //tensor operation is still executing asynchronously
          dag.registerExecutingNode(node,exec_handle);
        }
      }else{ //tensor operation not submitted due to either temporary resource shortage or fatal error
        auto discarded = this->node_executor_->discard(exec_handle);
        dag.setNodeIdle(node);
        issued = false;
        if(error_code != TRY_LATER){ //fatal error
          std::cout << "#ERROR(exatn::TensorRuntime::GraphExecutorLazy): Failed to submit tensor operation "
           << node << " with execution handle " << exec_handle << ": Error " << error_code << std::endl << std::flush;
          assert(false); //`Do I need to handle this case gracefully?
        }
      }
    }
    return issued;
  };

  auto test_nodes_for_completion = [this,&dag,&progress] () {
    auto executing_nodes = dag.executingNodesBegin();
    while(executing_nodes != dag.executingNodesEnd()){
      int error_code;
      auto exec_handle = executing_nodes->second;
      auto synced = this->node_executor_->sync(exec_handle,&error_code,false);
      if(synced){ //tensor operation has completed
        VertexIdType node;
        executing_nodes = dag.extractExecutingNode(executing_nodes,&node);
        auto & dag_node = dag.getNodeProperties(node);
        auto op = dag_node.getOperation();
        op->recordFinishTime();
        dag.setNodeExecuted(node,error_code);
        if(error_code == 0){
          dag.progressFrontNode(node);
          progress.front = dag.getFrontNode();
        }else{
          std::cout << "#ERROR(exatn::TensorRuntime::GraphExecutorLazy): Deferred completion error for tensor operation "
           << node << " with execution handle " << exec_handle << ": Error " << error_code << std::endl << std::flush;
          assert(false); //`Do I need to handle this case gracefully?
        }
      }else{ //tensor operation has not completed yet
        ++executing_nodes;
      }
    }
    return;
  };

  //`Finish

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
