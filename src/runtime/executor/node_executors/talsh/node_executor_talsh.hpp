/** ExaTN:: Tensor Runtime: Tensor graph node executor: Talsh
REVISION: 2019/07/24

Copyright (C) 2018-2019 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

#ifndef EXATN_RUNTIME_TALSH_NODE_EXECUTOR_HPP_
#define EXATN_RUNTIME_TALSH_NODE_EXECUTOR_HPP_

#include "tensor_node_executor.hpp"

namespace exatn {
namespace runtime {

class TalshNodeExecutor : public TensorNodeExecutor {

public:

  NodeExecHandleType execute(TensorOperation & op) override;

  bool sync(NodeExecHandleType op_handle, bool wait) override;

protected:
 //`TALSH executor state
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_TALSH_NODE_EXECUTOR_HPP_
