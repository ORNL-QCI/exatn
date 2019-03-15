/** ExaTN::Numerics: Space Basis
REVISION: 2019/03/15

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

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
