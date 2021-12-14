/** ExaTN: Tensor Runtime: Tensor network executor: NVIDIA cuQuantum
REVISION: 2021/12/14

Copyright (C) 2018-2021 Dmitry Lyakh
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

//#ifdef CUQUANTUM

#ifndef EXATN_RUNTIME_CUQUANTUM_EXECUTOR_HPP_
#define EXATN_RUNTIME_CUQUANTUM_EXECUTOR_HPP_

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

protected:

};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_CUQUANTUM_EXECUTOR_HPP_

//#endif //CUQUANTUM
