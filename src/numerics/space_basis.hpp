/** ExaTN::Numerics: Spaces/Subspaces
REVISION: 2019/01/27

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef SPACE_BASIS_HPP_
#define SPACE_BASIS_HPP_

#include "tensor_basic.hpp"

#include <assert.h>

#include <iostream>
#include <vector>

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

 /** Print. **/
 void printIt() const;

private:

 DimExtent basis_dim_;
 std::vector<BasisVector> basis_;
};

} //namespace numerics

} //namespace exatn

#endif //SPACES_BASIS_HPP_
