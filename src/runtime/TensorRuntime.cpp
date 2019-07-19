#include "TensorRuntime.hpp"
#include "exatn.hpp"

namespace exatn {
namespace runtime {

void TensorRuntime::openScope(const std::string &scopeName) {
  assert(scopeName.length() > 0);
  // complete the current scope first
  if(currentScope.length() > 0) closeScope();
  // create new graph with name given by scope name and store it in the dags map
  auto new_pos = dags.emplace(std::make_pair(
                               scopeName,exatn::getService<TensorGraph>("boost-digraph")
                              )
                             );
  assert(new_pos.second); // to make sure there was no other scope with the same name
  currentScope = scopeName; // change the name of the current scope
  return;
}


void TensorRuntime::pauseScope() {
  //`complete all currently executed tensor operations
  return;
}


void TensorRuntime::resumeScope(const std::string &scopeName) {
  assert(scopeName.length() > 0);
  // pause the current scope first
  if(currentScope.length() > 0) pauseScope();
  //`resume the execution of the previously paused scope
  currentScope = scopeName; // change the name of the current scope
  return;
}


void TensorRuntime::closeScope() {
  //`complete all operations in the current scope:
  currentScope = "";
  return;
}


void TensorRuntime::submit(std::shared_ptr<numerics::TensorOperation> op) {
  assert(currentScope.length() > 0);
  // upate the output tensor execution table
  auto newop_outid = op->getTensorOperandId(0);
  mtx.lock();
  auto curTableIter = outTensorExecTbl.find(currentScope);
  assert(curTableIter != outTensorExecTbl.end());
  auto &cur_table = curTableIter->second;
  if(cur_table.find(newop_outid) == cur_table.end())
    cur_table[newop_outid] = 1;
  else
    cur_table[newop_outid] += 1;

  // work on graph at dags[currentScope]
  // add on to the graph
  std::shared_ptr<TensorGraph> tg = dags[currentScope];
  auto tg_sz = tg->size();
  auto op1 = std::make_shared<TensorOpNode>(op);
  tg->addVertex(op1);
  auto num_operands = op->getNumOperands();
  std::shared_ptr<TensorOpNode> op0;
  for(int j=1; j<num_op1_operands; j++) {
    for(decltype(tg_sz) i = tg_sz-1; i >= 0; i--) {
      op0=tg->getVertexProperties(i);
      if(op0->op->getTensorOperandId(0) == op1->op->getTensorOperandId(j)) {
        tg->addEdge(op0,op1);
        break;
      } 
    } 
  }
  mtx.unlock();
  return;
}

bool TensorRuntime::sync(const numerics::TensorOperation &op, bool wait) {
  // sync on a particular tensor, everything related to tensor
  const auto op_outid = op.getTensorOperandId(0);
  mtx.lock();
  bool completed = (outTensorExecTbl[currentScope][op_outid] == 0);
  mtx.unlock();
  if(wait) {
    while(!completed) {
      mtx.lock();
      completed = (outTensorExecTbl[currentScope][op_outid] == 0);
      mtx.unlock();
    }
  }
  return completed;
}

bool TensorRuntime::sync(const numerics::Tensor &tensor, bool wait) {
  // sync on a particular tensor, everything related to tensor
  const auto op_outid = tensor.getTensorId();
  mtx.lock();
  bool completed = (outTensorExecTbl[currentScope][op_outid] == 0);
  mtx.unlock();
  if(wait) {
    while(!completed) {
      mtx.lock();
      completed = (outTensorExecTbl[currentScope][op_outid] == 0);
      mtx.unlock();
    }
  }
  return completed;
}

TensorDenseBlock TensorRuntime::getTensorData(const numerics::Tensor &tensor) {
  // sync
  assert(sync(tensor,true));
  // get tensor data after sync
  return TensorDenseBlock();
}

} // namespace runtime
} // namespace exatn
