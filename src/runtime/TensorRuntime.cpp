#include "TensorRuntime.hpp"
#include "exatn.hpp"

namespace exatn {
namespace runtime {

void TensorRuntime::openScope(const std::string &scopeName) {
  assert(scopeName.length() > 0);
  // complete the current scope first
  if(currentScope.length() > 0) closeScope();
  // create new graph with name given by scope name and store it in the dags map
  auto new_pos_dag = dags.emplace(std::make_pair(
                               scopeName,exatn::getService<TensorGraph>("boost-digraph")
                              )
                             );
  assert(new_pos_dag.second); // to make sure there was no other scope with the same name
  auto new_pos_tbl = outTensorExecTbl.emplace(std::make_pair(scopeName,std::map<std::size_t,int>{}));
  assert(new_pos_tbl.second); // to make sure there was no other scope with the same name
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
  if(currentScope.length() > 0){
    //`complete all operations in the current scope:
    assert(outTensorExecTbl.erase(currentScope) == 1);
    assert(dags.erase(currentScope) == 1);
    currentScope = "";
  }
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
  auto new_op_node = std::make_shared<TensorOpNode>(op);
  tg->addVertex(new_op_node);
  auto tg_sz = tg->size();
  auto num_operands = op->getNumOperands();
  for(int j = 1; j < num_operands; j++) {
    for(decltype(tg_sz) i = tg_sz-1; i >= 0; i--) {
      const auto & dag_node = tg->getVertexProperties(i);
      if(dag_node->op->getTensorOperandId(0) == new_op_node->op->getTensorOperandId(j)) {
        tg->addEdge(new_op_node,dag_node);
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
