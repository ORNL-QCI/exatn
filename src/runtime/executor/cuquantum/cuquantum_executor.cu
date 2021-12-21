/** ExaTN: Tensor Runtime: Tensor network executor: NVIDIA cuQuantum
REVISION: 2021/12/14

Copyright (C) 2018-2021 Dmitry Lyakh
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

#ifdef CUQUANTUM

#include "cuquantum_executor.hpp"

#include <iostream>

namespace exatn {
namespace runtime {

CuQuantumExecutor::CuQuantumExecutor()
{
 const size_t version = cutensornetGetVersion();
 std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): Version " << version << std::endl;
}

} //namespace runtime
} //namespace exatn

#endif //CUQUANTUM
