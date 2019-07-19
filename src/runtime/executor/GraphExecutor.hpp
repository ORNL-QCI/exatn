#ifndef EXATN_RUNTIME_GRAPH_EXECUTOR_HPP_
#define EXATN_RUNTIME_GRAPH_EXECUTOR_HPP_

#include "TensorGraph.hpp"

#include <iostream>
#include <memory>

namespace exatn {
namespace runtime {

// temp
using TensorOp = int;

class GraphExecutor {

public:

   void execute(TensorGraph& dag);

protected:

   virtual void exec_impl(TensorOp& op) = 0;

};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_GRAPH_EXECUTOR_HPP_
