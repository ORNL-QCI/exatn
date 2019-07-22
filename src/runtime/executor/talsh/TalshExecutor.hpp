#ifndef EXATN_RUNTIME_TALSH_EXECUTOR_HPP_
#define EXATN_RUNTIME_TALSH_EXECUTOR_HPP_

#include "GraphExecutor.hpp"

namespace exatn {
namespace runtime {

class TalshExecutor : public GraphExecutor {
protected:
   void exec_impl(numerics::TensorOperation& op) override;
};
}
}
#endif
