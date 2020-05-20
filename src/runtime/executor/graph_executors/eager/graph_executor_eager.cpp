/** ExaTN:: Tensor Runtime: Tensor graph executor: Eager
REVISION: 2020/05/20

Copyright (C) 2018-2020 Tiffany Mintz, Dmitry Lyakh, Alex McCaskey
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle)
**/

#include "graph_executor_eager.hpp"

#include "timers.hpp"

#include "talshxx.hpp"

#include <iostream>
#include <iomanip>

#include <cassert>

namespace exatn {
namespace runtime {

void EagerGraphExecutor::execute(TensorGraph & dag) {
  auto num_nodes = dag.getNumNodes();
  auto current = dag.getFrontNode();
  while(current < num_nodes){
    TensorOpExecHandle exec_handle;
    auto & dag_node = dag.getNodeProperties(current);
    if(!(dag_node.isExecuted())){
      dag.setNodeExecuting(current);
      auto op = dag_node.getOperation();
      if(logging_.load() != 0){
        logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR()
                 << "](EagerGraphExecutor)[EXEC_THREAD]: Submitting tensor operation "
                 << current << ": Opcode = " << static_cast<int>(op->getOpcode()); //debug
        if(logging_.load() > 1){
          logfile_ << ": Details:" << std::endl;
          op->printItFile(logfile_);
        }
      }
      op->recordStartTime();
      auto error_code = op->accept(*node_executor_,&exec_handle);
      if(logging_.load() != 0){
        logfile_ << ": Status = " << error_code << ": "; //debug
      }
      if(error_code == 0){
        if(logging_.load() != 0){
          logfile_ << "Syncing ... "; //debug
        }
        auto synced = node_executor_->sync(exec_handle,&error_code,true);
        op->recordFinishTime();
        if(synced && error_code == 0){
          dag.setNodeExecuted(current);
          if(logging_.load() != 0){
            logfile_ << "Success [" << std::fixed << std::setprecision(6)
                     << exatn::Timer::timeInSecHR() << "]" << std::endl; //debug
            logfile_.flush();
          }
          dag.progressFrontNode(current);
          ++current;
        }else{ //failed to synchronize the submitted tensor operation
          node_executor_->discard(exec_handle);
          if(error_code != 0) dag.setNodeExecuted(current,error_code);
          if(logging_.load() != 0){
            logfile_ << "Failed to synchronize tensor operation: Error " << error_code << std::endl;
            logfile_.flush();
          }
          std::cout << "#ERROR(exatn::TensorRuntime::GraphExecutorEager): Failed to synchronize tensor operation: Error "
                    << error_code << std::endl << std::flush;
          assert(false);
        }
      }else{ //failed to submit the tensor operation
        node_executor_->discard(exec_handle);
        dag.setNodeIdle(current);
        if(error_code != TRY_LATER && error_code != DEVICE_UNABLE){
         std::cout << "#ERROR(exatn::TensorRuntime::GraphExecutorEager): Failed to submit tensor operation: Error "
                   << error_code << std::endl << std::flush;
         assert(false);
        }
        if(logging_.load() != 0){
          logfile_ << "Will retry again" << std::endl; //debug
          logfile_.flush();
        }
      }
    }else{
      ++current;
    }
    num_nodes = dag.getNumNodes();
  }
  return;
}

} //namespace runtime
} //namespace exatn
