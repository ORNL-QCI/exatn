/** ExaTN: Quantum computing related
REVISION: 2021/06/28

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
**/

#ifndef EXATN_QUANTUM_HPP_
#define EXATN_QUANTUM_HPP_

#include "tensor_operator.hpp"
#include "tensor_symbol.hpp"

#include <vector>
#include <complex>
#include <string>

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

/** Creates a tensor network operator for a given spin Hamiltonian
    stored in a file (linear combination of Pauli strings). **/
std::shared_ptr<numerics::TensorOperator> readSpinHamiltonian(const std::string & operator_name,
                                                              const std::string & filename,
                                                              unsigned int num_sites);

} //namespace exatn

#endif //EXATN_QUANTUM_HPP_
