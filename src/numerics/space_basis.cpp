/** ExaTN::Numerics: Space Basis
REVISION: 2019/03/17

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "space_basis.hpp"

#include <iostream>

namespace exatn{

namespace numerics{

SpaceBasis::SpaceBasis(DimExtent space_dim):
 basis_dim_(space_dim)
{
}

void SpaceBasis::printIt() const
{
 std::cout << "SpaceBasis{Dim = " << basis_dim_ << "}";
 return;
}

DimExtent SpaceBasis::getDimension() const
{
 return basis_dim_;
}

} //namespace numerics

} //namespace exatn
