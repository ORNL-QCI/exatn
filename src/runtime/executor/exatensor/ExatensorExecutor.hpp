#ifndef EXATN_RUNTIME_EXATENSOR_EXECUTOR_HPP_
#define EXATN_RUNTIME_EXATENSOR_EXECUTOR_HPP_

#include "GraphExecutor.hpp"

namespace exatn {
namespace runtime {

class ExatensorExecutor : public GraphExecutor {
protected:
   void exec_impl(TensorOp& op) override;
};
}
}
#endif
