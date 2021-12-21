/** ExaTN: Tensor Runtime: Tensor network executor: NVIDIA cuQuantum
REVISION: 2021/12/20

Copyright (C) 2018-2021 Dmitry Lyakh
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 - ExaTN graph executor may accept whole tensor networks for execution
   via the optional cuQuantum backend in which case the graph executor
   will delegate execution of whole tensor networks to CuQuantumExecutor.

**/

#ifdef CUQUANTUM

#ifndef EXATN_RUNTIME_CUQUANTUM_EXECUTOR_HPP_
#define EXATN_RUNTIME_CUQUANTUM_EXECUTOR_HPP_

#include "tensor_network.hpp"
#include "tensor_operation.hpp"

#include <cutensornet.h>
#include <cutensor.h>
#include <cuda_runtime.h>

#include <unordered_map>
#include <vector>
#include <cassert>

#include "errors.hpp"

namespace exatn {
namespace runtime {

class CuQuantumExecutor {

public:

 CuQuantumExecutor();

 virtual ~CuQuantumExecutor() = default;

 int execute(numerics::TensorNetwork & network,
             TensorOpExecHandle * exec_handle);

 bool sync(TensorOpExecHandle op_handle,
           int * error_code,
           bool wait = true);

 bool sync();

protected:

};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_CUQUANTUM_EXECUTOR_HPP_

#endif //CUQUANTUM
