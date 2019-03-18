/** ExaTN::Numerics: Space Basis
REVISION: 2019/03/18

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "space_basis.hpp"

#include <iostream>
#include "assert.h"

namespace exatn{

namespace numerics{

SpaceBasis::SpaceBasis(DimExtent space_dim):
 basis_dim_(space_dim)
{
}

SpaceBasis::SpaceBasis(DimExtent space_dim,
                       const std::vector<SymmetryRange> & symmetry_subranges):
 basis_dim_(space_dim)
{
 for(const auto & subrange: symmetry_subranges) this->registerSymmetrySubrange(subrange);
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

const std::vector<SymmetryRange> & SpaceBasis::getSymmetrySubranges() const
{
 return symmetry_ranges_;
}

void SpaceBasis::registerSymmetrySubrange(const SymmetryRange subrange)
{
 assert(subrange.upper < basis_dim_ && subrange.lower <= subrange.upper);
 symmetry_ranges_.emplace_back(subrange);
 return;
}

} //namespace numerics

} //namespace exatn
