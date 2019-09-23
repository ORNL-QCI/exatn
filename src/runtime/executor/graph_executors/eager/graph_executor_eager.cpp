/** ExaTN:: Tensor Runtime: Tensor graph executor: Eager
REVISION: 2019/09/23

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
      std::cout << "#DEBUG(EagerGraphExecutor)[EXEC_THREAD]: Submitting tensor operation " << current; //debug
      auto error_code = dag_node.getOperation()->accept(*node_executor_,&exec_handle);
      std::cout << ": Status = " << error_code << ": "; //debug
      if(error_code == 0){
        std::cout << "Syncing ... "; //debug
        auto synced = node_executor_->sync(exec_handle,&error_code,true);
        if(synced && error_code == 0){
          dag.setNodeExecuted(current);
          std::cout << "Success" << std::endl; //debug
          ++current;
        }else{
          if(error_code != 0) dag.setNodeExecuted(current,error_code);
          std::cout << "Failed to synchronize tensor operation: Error " << error_code << std::endl;
          assert(false);
        }
      }else{
        dag.setNodeIdle(current);
        std::cout << "Will retry again" << std::endl; //debug
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
