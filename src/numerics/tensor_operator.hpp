/** ExaTN::Numerics: Tensor operator
REVISION: 2019/10/10

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) A tensor network vector is a vector in a given tensor space with
     its expansion (tensor) coefficients factorized as a tensor network.
     A ket tensor network vector produces its corresponding dual bra
     tensor network vector upon complex conjugation of all constituting
     tensor factors and reversing the direction of the output tensor legs.
 (b) A tensor operator is an ordered linear combination of tensors and
     tensor networks in which the output tensor legs are distinguished
     as bra and ket tensor legs: The bra tensor legs contract with legs
     of a bra tensor network vector, the ket tensor legs contract with
     legs of a ket tensor network vector.
 (c) The first component of the tensor operator is applied first when
     acting on a ket vector. The last component of the tensor operator
     is applied first when acting on a bra vector.
 (d) The order of components of a tensor operator is reversed upon conjugation.
**/

#ifndef EXATN_NUMERICS_TENSOR_OPERATOR_HPP_
#define EXATN_NUMERICS_TENSOR_OPERATOR_HPP_

#include "tensor_basic.hpp"
#include "tensor_network.hpp"

#include <string>
#include <vector>
#include <complex>
#include <memory>

namespace exatn{

namespace numerics{

class TensorOperator{
public:

 //Tensor operator component:
 struct OperatorComponent{
  //Tensor network (or a single tensor stored as a tensor network of size 1):
  std::shared_ptr<TensorNetwork> network;
  //Ket legs of the tensor network: Output tensor leg --> global tensor mode id:
  std::vector<std::pair<unsigned int, unsigned int>> ket_legs;
  //Bra legs of the tensor network: Output tensor leg --> global tensor mode id:
  std::vector<std::pair<unsigned int, unsigned int>> bra_legs;
  //Expansion coefficient of the operator component:
  std::complex<double> coefficient;
 };

 using Iterator = typename std::vector<OperatorComponent>::iterator;
 using ConstIterator = typename std::vector<OperatorComponent>::const_iterator;

 TensorOperator(const std::string & name): name_(name) {}

 TensorOperator(const TensorOperator &) = default;
 TensorOperator & operator=(const TensorOperator &) = default;
 TensorOperator(TensorOperator &&) noexcept = default;
 TensorOperator & operator=(TensorOperator &&) noexcept = default;
 virtual ~TensorOperator() = default;

 inline Iterator begin() {return components_.begin();}
 inline Iterator end() {return components_.end();}
 inline ConstIterator cbegin() {return components_.cbegin();}
 inline ConstIterator cend() {return components_.cend();}

 /** Returns the total number of components in the tensor operator. **/
 std::size_t getNumComponents() const{
  return components_.size();
 }

 /** Returns a specific component of the tensor operator. **/
 const OperatorComponent & getComponent(std::size_t component_num){
  assert(component_num < components_.size());
  return components_[component_num];
 }

 /** Appends a new component to the tensor operator linear expansion. The new component
     can either be a tensor network or just a single tensor expressed as a tensor network
     of size 1. The ket and bra pairing arguments specify which legs of the network output
     tensor act on a ket vector and which on a bra vector, together with their mapping onto
     the global modes of the tensor space the tensor operator is supposed to act upon. **/
 bool appendComponent(std::shared_ptr<TensorNetwork> network,                                 //in: tensor network (or single tensor as a tensor network)
                      const std::vector<std::pair<unsigned int, unsigned int>> & ket_pairing, //in: ket pairing: Output tensor leg --> global tensor mode id
                      const std::vector<std::pair<unsigned int, unsigned int>> & bra_pairing, //in: bra pairing: Output tensor leg --> global tensor mode id
                      const std::complex<double> coefficient);                                //in: expansion coefficient

protected:

 std::string name_;                          //tensor operator name
 std::vector<OperatorComponent> components_; //ordered components of the tensor operator
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OPERATOR_HPP_
