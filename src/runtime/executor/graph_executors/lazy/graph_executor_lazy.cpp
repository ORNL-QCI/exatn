/** ExaTN:: Tensor Runtime: Tensor graph executor: Lazy
REVISION: 2020/09/02

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
    VertexIdType num_nodes; //total number of nodes in the DAG (may grow)
    VertexIdType front;     //the first unexecuted node in the DAG
    VertexIdType current;   //the current node in the DAG
  };

  Progress progress{dag.getNumNodes(),dag.getFrontNode(),0};
  progress.current = progress.front;

  auto find_next_idle_node = [this,&dag,&progress] () {
    const auto prev_node = progress.current;
    progress.front = dag.getFrontNode();
    progress.num_nodes = dag.getNumNodes();
    if(progress.front < progress.num_nodes){
      ++progress.current;
      if(progress.current >= progress.num_nodes){
        progress.current = progress.front;
        if(progress.current == prev_node) ++progress.current;
      }else{
        if(progress.current >= (progress.front + this->getPipelineDepth())){
          progress.current = progress.front;
          if(progress.current == prev_node) ++progress.current;
        }
      }
      while(progress.current < progress.num_nodes){
        if(dag.nodeIdle(progress.current)) break;
        ++progress.current;
      }
    }else{ //all DAG nodes have been executed
      progress.current = progress.front; //end-of-DAG
    }
    progress.num_nodes = dag.getNumNodes(); //update DAG size again
    return (progress.current < progress.num_nodes && progress.current != prev_node);
  };

  auto inspect_node_dependencies = [this,&dag,&progress] () {
    bool ready_for_execution = false;
    if(progress.current < progress.num_nodes){
      auto & dag_node = dag.getNodeProperties(progress.current);
      ready_for_execution = dag_node.isIdle();
      if(ready_for_execution){ //node is idle
        ready_for_execution = ready_for_execution && dag.nodeDependenciesResolved(progress.current);
        if(ready_for_execution){ //all node dependencies resolved (or none)
          auto registered = dag.registerDependencyFreeNode(progress.current);
          if(registered && logging_.load() > 1) logfile_ << "DAG node detected with all dependencies resolved: " << progress.current << std::endl;
        }else{ //node still has unresolved dependencies, try prefetching
          if(progress.current < (progress.front + this->getPrefetchDepth())){
            auto prefetching = this->node_executor_->prefetch(*(dag_node.getOperation()));
            if(logging_.load() != 0 && prefetching){
              logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
                       << "](LazyGraphExecutor)[EXEC_THREAD]: Initiated prefetch for tensor operation "
                       << progress.current << std::endl;
#ifndef NDEBUG
              logfile_.flush();
#endif
            }
          }
        }
      }
    }
    return ready_for_execution;
  };

  auto issue_ready_node = [this,&dag,&progress] () {
    if(logging_.load() > 2){
      logfile_ << "DAG current list of dependency free nodes:";
      auto free_nodes = dag.getDependencyFreeNodes();
      for(const auto & node: free_nodes) logfile_ << " " << node;
      logfile_ << std::endl;
    }
    VertexIdType node;
    bool issued = dag.extractDependencyFreeNode(&node);
    if(issued){
      auto & dag_node = dag.getNodeProperties(node);
      auto op = dag_node.getOperation();
      if(logging_.load() != 0){
        logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
                 << "](LazyGraphExecutor)[EXEC_THREAD]: Submitting tensor operation "
                 << node << ": Opcode = " << static_cast<int>(op->getOpcode());
        if(logging_.load() > 1){
          logfile_ << ": Details:" << std::endl;
          op->printItFile(logfile_);
        }
#ifndef NDEBUG
        logfile_.flush();
#endif
      }
      dag.setNodeExecuting(node);
      op->recordStartTime();
      TensorOpExecHandle exec_handle;
      auto error_code = op->accept(*(this->node_executor_),&exec_handle);
      if(logging_.load() != 0) logfile_ << ": Status = " << error_code;
      if(error_code == 0){ //tensor operation submitted for execution successfully
        if(logging_.load() != 0) logfile_ << ": Syncing ... ";
        auto synced = this->node_executor_->sync(exec_handle,&error_code,false);
        if(synced){ //tensor operation has completed immediately
          op->recordFinishTime();
          dag.setNodeExecuted(node,error_code);
          if(error_code == 0){
            if(logging_.load() != 0){
              logfile_ << "Success [" << std::fixed << std::setprecision(6)
                       << exatn::Timer::timeInSecHR(getTimeStampStart()) << "]" << std::endl;
#ifndef NDEBUG
              logfile_.flush();
#endif
            }
            progress.num_nodes = dag.getNumNodes();
            auto progressed = dag.progressFrontNode(node);
            if(progressed){
              progress.front = dag.getFrontNode();
              while(progress.front < progress.num_nodes){
                if(!(dag.nodeExecuted(progress.front))) break;
                dag.progressFrontNode(progress.front);
                progress.front = dag.getFrontNode();
              }
            }
            if(progressed && logging_.load() > 1) logfile_ << "DAG front node progressed to "
              << progress.front << " out of total of " << progress.num_nodes << std::endl;
          }else{
            if(logging_.load() != 0){
              logfile_ << "Failed: Error " << error_code << " [" << std::fixed << std::setprecision(6)
                       << exatn::Timer::timeInSecHR(getTimeStampStart()) << "]" << std::endl;
              logfile_.flush();
            }
            std::cout << "#ERROR(exatn::TensorRuntime::GraphExecutorLazy): Immediate completion error for tensor operation "
             << node << " with execution handle " << exec_handle << ": Error " << error_code << std::endl << std::flush;
            assert(false); //`Do I need to handle this case gracefully?
          }
        }else{ //tensor operation is still executing asynchronously
          dag.registerExecutingNode(node,exec_handle);
          if(logging_.load() != 0) logfile_ << "Deferred" << std::endl;
        }
      }else{ //tensor operation not submitted due to either temporary resource shortage or fatal error
        auto discarded = this->node_executor_->discard(exec_handle);
        dag.setNodeIdle(node);
        auto registered = dag.registerDependencyFreeNode(node); assert(registered);
        issued = false;
        if(error_code == TRY_LATER){ //temporary shortage of resources
          if(logging_.load() != 0) logfile_ << ": Postponed" << std::endl;
        }else{ //fatal error
          if(logging_.load() != 0) logfile_.flush();
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
          if(logging_.load() != 0){
            logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
                     << "](LazyGraphExecutor)[EXEC_THREAD]: Synced tensor operation "
                     << node << ": Opcode = " << static_cast<int>(op->getOpcode()) << std::endl;
#ifndef NDEBUG
            logfile_.flush();
#endif
          }
          progress.num_nodes = dag.getNumNodes();
          auto progressed = dag.progressFrontNode(node);
          if(progressed){
            progress.front = dag.getFrontNode();
            while(progress.front < progress.num_nodes){
              if(!(dag.nodeExecuted(progress.front))) break;
              dag.progressFrontNode(progress.front);
              progress.front = dag.getFrontNode();
            }
          }
          if(progressed && logging_.load() > 1) logfile_ << "DAG front node progressed to "
            << progress.front << " out of total of " << progress.num_nodes << std::endl;
        }else{
          if(logging_.load() != 0){
            logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
                     << "](LazyGraphExecutor)[EXEC_THREAD]: Failed to sync tensor operation "
                     << node << ": Opcode = " << static_cast<int>(op->getOpcode()) << std::endl;
            logfile_.flush();
          }
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

  if(logging_.load() != 0){
    logfile_ << "DAG entry list of dependency free nodes:";
    auto free_nodes = dag.getDependencyFreeNodes();
    for(const auto & node: free_nodes) logfile_ << " " << node;
    logfile_ << std::endl << std::flush;
  }
  bool not_done = (progress.front < progress.num_nodes);
  while(not_done){
    //Try to issue all idle DAG nodes that are ready for execution:
    while(issue_ready_node());
    //Inspect whether the current node can be issued:
    auto node_ready = inspect_node_dependencies();
    //Test the currently executing DAG nodes for completion:
    test_nodes_for_completion();
    //Find the next idle DAG node:
    not_done = find_next_idle_node() || (progress.front < progress.num_nodes);
  }
  return;
}

} //namespace runtime
} //namespace exatn
