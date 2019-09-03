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
    dag.lock();
    dag.setNodeExecuting(current);
    std::cout << "#DEBUG(EagerGraphExecutor)[EXEC_THREAD]: Submitting tensor operation " << current; //debug
    auto error_code = dag.getNodeProperties(current).getOperation()->accept(*node_executor_,&exec_handle);
    dag.unlock();
    std::cout << ": Status = " << error_code << ": "; //debug
    if(error_code == 0){
      std::cout << "Syncing ... "; //debug
      auto synced = node_executor_->sync(exec_handle,&error_code,true);
      if(synced && error_code == 0){
        dag.setNodeExecuted(current);
        std::cout << "Success" << std::endl; //debug
        ++current;
      }else{
        std::cout << "Failure to synchronize tensor operation: Error " << error_code << std::endl;
        assert(false);
      }
    }else{
      std::cout << "Will retry" << std::endl; //debug
    }
    num_nodes = dag.getNumNodes();
  }
  return;
}

} //namespace runtime
} //namespace exatn
