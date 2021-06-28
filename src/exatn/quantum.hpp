/** ExaTN: Quantum computing related
REVISION: 2021/06/28

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
**/

#ifndef EXATN_QUANTUM_HPP_
#define EXATN_QUANTUM_HPP_

#include <vector>
#include <complex>

#include "errors.hpp"

namespace exatn{

enum class QuantumGate{
 gate_I,
 gate_X,
 gate_Y,
 gate_Z,
 gate_H,
 gate_S,
 gate_T
};

/** Returns the data initialization vector for a specific quantum gate
    that can subsequently be used for initializing its tensor. **/
const std::vector<std::complex<double>> & getQuantumGateData(const QuantumGate gate_name,
                                                             std::initializer_list<double> angles);

} //namespace exatn

#endif //EXATN_QUANTUM_HPP_
