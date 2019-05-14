#ifndef EXATN_RUNTIME_DAGOPT_HPP_
#define EXATN_RUNTIME_DAGOPT_HPP_

#include <iostream>
#include <memory>

namespace exatn {
namespace runtime {

// temp
using TensorOp = int;

class GraphExecutor {

public:

   virtual void execute(Graph& dag);
   
protected:

   virtual void exec_impl(TensorOp& op) = 0;

};
}
} // namespace exatn
#endif
