#include "TensorRuntime.hpp"
#include "TensorGraph.hpp"

#include<memory>

namespace exatn {
namespace runtime {
void TensorRuntime::openScope(const std::string &scopeName) {
  assert(scopeName.length() > 0);
  // save currentScope name
  currentScope = scopeName;
  // create new graph with name given by scope name
  // store it in the dags map
  dags[currentScope] = std::make_shared<TensorGraph>();
}

void TensorRuntime::closeScope() { currentScope = ""; }

void TensorRuntime::submit(std::shared_ptr<TensorOperation> op) {
  //Call sync on a single operation for now
  sync(op);

  // work on graph at dags[currentScope]
  // add on to the graph
  std::shared_ptr<TensorGraph> tg=dags[currentScope];
  int tg_sz=tg->size();
  std::shared_ptr<TensorOpNode> op1=std::make_shared<TensorOpNode>(op);
  tg->addVertex(op1);
  unsigned int num_op1_operands = op->getNumOperands();
  TensorOpNode op0;
  bool no_edge=true;
  for(int i=tg_sz-1; i>=0; i--)
  {
	op0=tg->getVertexProperties(i);
	std::size_t op0_outid = op0.op->getTensorOperandId(0);
	for(int j=1; j<num_op1_operands; j++) {
		if(op0_outid == op1->op->getTensorOperandId(j))
		{
			tg->addEdge(op0,op1);
			no_edge=false;
		}
	}
  }
	//add edge to dummy node if no edge added (may not be necessary)
}

void TensorRuntime::sync(const std::shared_ptr<TensorOperation> &op) {
  // sync on a particular tensor, everything related to tensor
  // must complete
}

void TensorRuntime::sync(const exatn::numerics::Tensor &tensor) {
  // sync on a particular tensor, everything related to tensor
  // must complete
}

TensorDenseBlock
TensorRuntime::getTensorData(const exatn::numerics::Tensor &tensor) {
  // get tensor data after sync
  return TensorDenseBlock();
}
} // namespace runtime
} // namespace exatn
