/** ExaTN::Numerics: Tensor network expansion
REVISION: 2019/10/25

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) A tensor network expansion is an ordered linear expansion
     consisting of tensor networks with complex coefficients.
     The output tensors of all constituting tensor networks must
     be congruent, that is, have the same shape and leg direction.
     The tensor network expansion is essentially a linear combination
     of tensor network vectors in a given tensor space. The rank of
     the tensor network expansion is the rank of the output tensors
     of all constituting tensor networks.
 (b) A tensor network expansion can either be a ket or a bra.
 (c) An inner product tensor network expansion can be formed by contracting
     one tensor network expansion with another tensor network expansion
     from the dual vector space (bra*ket, ket*bra).
 (d) A direct product tensor network expansion can be formed from
     two tensor network expansions from the same space (bra*bra, ket*ket).
 (e) A tensor network operator can be applied to a tensor network expansion,
     producing another tensor network expansion in the same space.
**/

#ifndef EXATN_NUMERICS_TENSOR_EXPANSION_HPP_
#define EXATN_NUMERICS_TENSOR_EXPANSION_HPP_

#include "tensor_basic.hpp"
#include "tensor_network.hpp"
#include "tensor_operator.hpp"

#include <string>
#include <vector>
#include <complex>
#include <memory>

namespace exatn{

namespace numerics{

class TensorExpansion{
public:

 //Tensor network expansion component:
 struct ExpansionComponent{
  //Tensor network:
  std::shared_ptr<TensorNetwork> network;
  //Expansion coefficient:
  std::complex<double> coefficient;
 };

 using Iterator = typename std::vector<ExpansionComponent>::iterator;
 using ConstIterator = typename std::vector<ExpansionComponent>::const_iterator;

 /** Constructs an empty ket tensor expansion. **/
 TensorExpansion(): ket_(true) {}

 /** Constructs a tensor expansion by applying a tensor network operator
     to another tensor network expansion. **/
 TensorExpansion(const TensorExpansion & expansion,       //in: tensor network expansion in some tensor space
                 const TensorOperator & tensor_operator); //in: tensor network operator

 /** Either constructs the inner product tensor network expansion by closing
     one tensor network expansion with another tensor network expansion
     from the dual tensor space or constructs the direct product tensor network
     expansion from two tensor network expansions from the same space. **/
 TensorExpansion(const TensorExpansion & left_expansion,   //in: tensor network expansion in some tensor space
                 const TensorExpansion & right_expansion); //in: tensor network expansion from the same or dual space

 /** Constructs the inner product tensor network expansion by applying
     a tensor network operator to a tensor network expansion (right_expansion)
     and then closing the resulting tensor network expansion with another
     tensor network expansion from the dual tensor space. **/
 TensorExpansion(const TensorExpansion & left_expansion,  //in: tensor network expansion in some tensor space
                 const TensorExpansion & right_expansion, //in: tensor network expansion from the dual tensor space
                 const TensorOperator & tensor_operator); //in: tensor network operator

 TensorExpansion(const TensorExpansion &) = default;
 TensorExpansion & operator=(const TensorExpansion &) = default;
 TensorExpansion(TensorExpansion &&) noexcept = default;
 TensorExpansion & operator=(TensorExpansion &&) noexcept = default;
 virtual ~TensorExpansion() = default;

 inline Iterator begin() {return components_.begin();}
 inline Iterator end() {return components_.end();}
 inline ConstIterator cbegin() {return components_.cbegin();}
 inline ConstIterator cend() {return components_.cend();}

 /** Returns the total number of components in the tensor network expansion. **/
 inline std::size_t getNumComponents() const{
  return components_.size();
 }

 /** Returns a specific component of the tensor network expansion. **/
 inline const ExpansionComponent & getComponent(std::size_t component_num){
  assert(component_num < components_.size());
  return components_[component_num];
 }

 /** Appends a new component to the tensor network expansion. **/
 bool appendComponent(std::shared_ptr<TensorNetwork> network,  //in: tensor network
                      const std::complex<double> coefficient); //in: expansion coefficient

 /** Returns whether the tensor network expansion is ket or not. **/
 inline bool isKet() const{
  return ket_;
 }

 inline bool isBra() const{
  return !ket_;
 }

 /** Conjugates the tensor network expansion: All constituting tensors are complex conjugated,
     all tensor legs reverse their direction, complex linear expansion coefficients are conjugated:
     The ket tensor network expansion becomes a bra, and vice versa. **/
 void conjugate();

protected:

 /** Internal methods: **/
 void constructDirectProductTensorExpansion(const TensorExpansion & left_expansion,
                                            const TensorExpansion & right_expansion);
 void constructInnerProductTensorExpansion(const TensorExpansion & left_expansion,
                                           const TensorExpansion & right_expansion);

 /** Data members: **/
 std::vector<ExpansionComponent> components_; //ordered components of the tensor network expansion
 bool ket_; //ket or bra
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_EXPANSION_HPP_
