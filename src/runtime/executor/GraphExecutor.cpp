#include "GraphExecutor.hpp"

namespace exatn {
namespace runtime {

void GraphExecutor::execute(TensorGraph& dag) {
  int nodes_executed=0;
  auto num_nodes = dag->order();
  while(nodes_executed <= num_nodes)
    exec_impl(nextExecutableNode(dag,nodes_executed));
}


numerics::TensorOperation GraphExecutor::nextExecutableNode(TensorGraph& dag, int &nodes_executed){
  auto num_nodes = dag->order();
  decltype(num_nodes) i; 
  for(i = 0; i < num_nodes; i++) {
    if(!dag->nodeExecuted(i)) {
      if(dag->degree(i)==0) {
        dag->setNodeExecuted(i);
        nodes_executed++;
        break;
      }
      else {
        auto n_list = dag->getNeighborList(i);
        int j;
        for(j=0; j<n_list.size(); j++)
          if(!dag->nodeExecuted(j))
            break;
        if(j>=n_list.size()) {
          dag->setNodeExecuted(i);
          nodes_executed++;
          break;
        }
      }
    }
  } 
  
  assert(i < num_nodes);

  return dag->getVertexProperties(i)->op;
}

}
}
