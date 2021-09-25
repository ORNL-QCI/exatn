/** ExaTN: Quantum computing related
REVISION: 2021/09/25

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 a) Provides utilities related to quantum circuit simulations, like quantum gates,
    Pauli matrix based Hamiltonian reading, etc.
 b) Normal gate action translates to the following tensor notation:
     q(i0) * G(i0|i1) --> v(i1),
     q(i0) q(i1) * G(i0,i1|i2,i3) --> v(i2,i3), etc.
    Note that if G(i0,i1|i2,i3) is a controlled 2-body gate,
    the control (senior) index is i1, not i0.
    A convenient way to remember this is to use the standard
    bra-ket convention such that we will have:
    |v(i3,i2)> = CX(i3,i2|i1,i0) * |q(i1)q(i0)>, where
    tensor leg numeration is from right to left:
      (i3)--V  =  (i3)--CX--(i1)--q
      (i2)--V     (i2)--CX--(i0)--q
    where the control line is i1.
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
    represented as a linear combination of Pauli strings. Supported formats:
    + "OpenFermion": Open Fermion format (default);
    + "QCWare": QCWare collab format (by Rob Parrish);
**/
std::shared_ptr<exatn::numerics::TensorOperator> readSpinHamiltonian(const std::string & operator_name,
                                                                     const std::string & filename,
                                                                     TensorElementType precision = TensorElementType::COMPLEX64,
                                                                     const std::string & format = "OpenFermion");
} //namespace quantum

} //namespace exatn

#endif //EXATN_QUANTUM_HPP_
