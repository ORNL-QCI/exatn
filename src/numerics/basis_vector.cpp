/** ExaTN::Numerics: Basis Vector
REVISION: 2019/03/17

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "basis_vector.hpp"

#include <iostream>

namespace exatn{

namespace numerics{

BasisVector::BasisVector(SubspaceId id):
 id_(id)
{
}

void BasisVector::printIt() const
{
 std::cout << "BasisVector{id = " << id_ << "}";
 return;
}

} //namespace numerics

} //namespace exatn
