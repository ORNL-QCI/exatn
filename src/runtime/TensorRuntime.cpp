#include "TensorRuntime.hpp"

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

void TensorRuntime::submit(std::shared_ptr<numerics::TensorOperation> op) {
  //upate the output tensor executation table
  int newop_outid = op->getTensorOperandId(0);
  std::map<std::string, std::map<int, int>>::iterator curTableIter;
  std::map<int, int>::iterator &cur_table;
  mtx.lock();
  curTableIter = outTensorExecTbl.find(currentScope);
  if(curTableIter != outTensorExecTbl.end())
  {
	cur_table = curTableIter.second;
	if(cur_table.find(newop_outid)==cur_table.end())
		cur_table[newop_outid]=1;
	else
		cur_table[newop_outid]+=1;
  }
  else
	outTensorExeciTbl[currentScope][newop_outid]=1;

  // work on graph at dags[currentScope]
  // add on to the graph
  std::shared_ptr<TensorGraph> tg=dags[currentScope];
  int tg_sz=tg->size();
  std::shared_ptr<TensorOpNode> op1=std::make_shared<TensorOpNode>(op);
  tg->addVertex(op1);
  unsigned int num_op1_operands = op->getNumOperands();
  TensorOpNode op0;
  for(int i=tg_sz-1; i>=0; i--)
  {
	op0=tg->getVertexProperties(i);
	std::size_t op0_outid = op0.op->getTensorOperandId(0);
	for(int j=1; j<num_op1_operands; j++) {
		if(op0_outid == op1->op->getTensorOperandId(j))
			tg->addEdge(op0,op1);
	}
  }
  mtx.unlock();
}

void TensorRuntime::sync(const std::shared_ptr<numerics::TensorOperation> &op) {
  // sync on a particular tensor, everything related to tensor
  bool syncing=true;
  int op_outid = op->getTensorOperandId(0);
  while(syncing)
  {
	mtx.lock();
		if(outTensorExec[op_outid]==0)
			syncing=false;
	mtx.unlock();
  }
}

void TensorRuntime::sync(const exatn::numerics::Tensor &tensor) {
  // sync on a particular tensor, everything related to tensor
  bool syncing=true;
  int tid = tensor.getTensorId();;
  while(syncing)
  {
        mtx.lock();
                if(outTensorExec[tid]==0)
                        syncing=false;
        mtx.unlock();
  }
}

TensorDenseBlock
TensorRuntime::getTensorData(const exatn::numerics::Tensor &tensor) {
  // get tensor data after sync
  return TensorDenseBlock();
}
} // namespace runtime
} // namespace exatn
