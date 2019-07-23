#include "GraphExecutor.hpp"

namespace exatn {
namespace runtime {

void GraphExecutor::execute(TensorGraph & dag) {
  int nodes_executed=0, execnode_id;
  auto num_nodes = dag.getNumNodes();

  while(nodes_executed <= num_nodes) {
    execnode_id = nextExecutableNodeId(dag);
    exec_impl(*((dag.getNodeProperties(execnode_id)).getOperation()));
    mtx.lock();
    //TODO: update output tensor execution table
    dag.setNodeExecuted(execnode_id);
    mtx.unlock();
    nodes_executed++;
    num_nodes = dag.getNumNodes();
  }
}

int GraphExecutor::nextExecutableNodeId(TensorGraph & dag){
  auto num_nodes = dag.getNumNodes();
  int i;
  for(i = 0; i < num_nodes; i++) {
    if(!dag.nodeExecuted(i)) {
      if(dag.degree(i)==0)
        break;
      else {
        auto n_list = dag.getNeighborList(i);
        int j;
        for(j = 0; j < n_list.size(); j++)
          if(!dag.nodeExecuted(j))
            break;
        if(j >= n_list.size())
          break;
      }
    }
  }

  assert(i < num_nodes);

  return i;
}

} //namespace runtime
} //namespace exatn
