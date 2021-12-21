/** ExaTN: Tensor Runtime: Tensor network executor: NVIDIA cuQuantum
REVISION: 2021/12/21

Copyright (C) 2018-2021 Dmitry Lyakh
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

#ifdef CUQUANTUM

#include <cutensornet.h>
#include <cutensor.h>
#include <cuda_runtime.h>

#include <vector>

#include <iostream>

#include "cuquantum_executor.hpp"

namespace exatn {
namespace runtime {

struct TensorNetworkReq {
 std::shared_ptr<numerics::TensorNetwork> network;
};


CuQuantumExecutor::CuQuantumExecutor()
{
 const size_t version = cutensornetGetVersion();
 std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): Version " << version << std::endl;
}

} //namespace runtime
} //namespace exatn

#endif //CUQUANTUM
