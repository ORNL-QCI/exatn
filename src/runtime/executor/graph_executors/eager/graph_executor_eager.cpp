/** ExaTN:: Tensor Runtime: Tensor graph executor: Eager
REVISION: 2019/10/13

Copyright (C) 2018-2019 Tiffany Mintz, Dmitry Lyakh, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)
**/

#include "graph_executor_eager.hpp"

#include <iostream>

#include <cassert>

namespace exatn {
namespace runtime {

void EagerGraphExecutor::execute(TensorGraph & dag) {
  auto num_nodes = dag.getNumNodes();
  decltype(num_nodes) current = 0;
  while(current < num_nodes){
    TensorOpExecHandle exec_handle;
    auto & dag_node = dag.getNodeProperties(current);
    if(!(dag_node.isExecuted())){
      dag.setNodeExecuting(current);
      auto op = dag_node.getOperation();
      if(logging_.load() != 0){
        logfile_ << "#DEBUG(EagerGraphExecutor)[EXEC_THREAD]: Submitting tensor operation "
                 << current << ": Opcode = " << static_cast<int>(op->getOpcode()); //debug
        if(logging_.load() > 1){
          logfile_ << ": Details:" << std::endl;
          op->printItFile(logfile_);
        }
      }
      auto error_code = op->accept(*node_executor_,&exec_handle);
      if(logging_.load() != 0){
        logfile_ << ": Status = " << error_code << ": "; //debug
      }
      if(error_code == 0){
        if(logging_.load() != 0){
          logfile_ << "Syncing ... "; //debug
        }
        auto synced = node_executor_->sync(exec_handle,&error_code,true);
        if(synced && error_code == 0){
          dag.setNodeExecuted(current);
          if(logging_.load() != 0){
            logfile_ << "Success" << std::endl; //debug
            logfile_.flush();
          }
          ++current;
        }else{
          if(error_code != 0) dag.setNodeExecuted(current,error_code);
          if(logging_.load() != 0){
            logfile_ << "Failed to synchronize tensor operation: Error " << error_code << std::endl;
            logfile_.flush();
          }
          std::cout << "Failed to synchronize tensor operation: Error " << error_code << std::endl;
          assert(false);
        }
      }else{
        dag.setNodeIdle(current);
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
