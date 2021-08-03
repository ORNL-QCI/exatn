/** ExaTN: Quantum computing related
REVISION: 2021/08/03

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
**/

#ifndef EXATN_QUANTUM_HPP_
#define EXATN_QUANTUM_HPP_

#include "exatn_numerics.hpp"
#include "tensor_operator.hpp"
#include "tensor_symbol.hpp"

#include <vector>
#include <complex>
#include <string>

#include "errors.hpp"

namespace exatn{

namespace quantum{

enum class Gate{
 gate_0,
 gate_I,
 gate_X,
 gate_Y,
 gate_Z,
 gate_H,
 gate_S,
 gate_T,
 gate_CX,
 gate_CY,
 gate_CZ,
 gate_SWAP,
 gate_ISWAP
};

/** Returns the data initialization vector for a specific quantum gate
    that can subsequently be used for initializing its tensor. **/
std::vector<std::complex<double>> getGateData(const Gate gate_name,
                                              std::initializer_list<double> angles = {});

/** Creates a tensor network operator for a given spin Hamiltonian
    stored in an OpenFermion file (linear combination of Pauli strings). **/
std::shared_ptr<exatn::numerics::TensorOperator> readSpinHamiltonian(const std::string & operator_name,
                                                                     const std::string & filename,
                                                                     TensorElementType precision = TensorElementType::COMPLEX64);
} //namespace quantum

} //namespace exatn

#endif //EXATN_QUANTUM_HPP_
