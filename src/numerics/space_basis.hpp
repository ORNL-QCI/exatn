/** ExaTN::Numerics: Space Basis
REVISION: 2019/03/17

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Space basis is a collection of linear-independent basis vectors.
 (b) An abstract space basis can be further specialized by storing
     specialized basis vectors with additional attributes. By default
     an abstract space basis is only characterized by its dimension.
**/

#ifndef SPACE_BASIS_HPP_
#define SPACE_BASIS_HPP_

#include "tensor_basic.hpp"
#include "basis_vector.hpp"

namespace exatn{

namespace numerics{

class SpaceBasis{
public:

 SpaceBasis(DimExtent space_dim);

 SpaceBasis(const SpaceBasis & space_basis) = default;
 SpaceBasis & operator=(const SpaceBasis & space_basis) = default;
 SpaceBasis(SpaceBasis && space_basis) = default;
 SpaceBasis & operator=(SpaceBasis && space_basis) = default;
 virtual ~SpaceBasis() = default;

 /** Prints. **/
 void printIt() const;

 /** Returns the basis dimension. **/
 DimExtent getDimension() const;

private:

 DimExtent basis_dim_; //basis dimension
};

} //namespace numerics

} //namespace exatn

#endif //SPACES_BASIS_HPP_
