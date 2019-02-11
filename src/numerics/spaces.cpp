/** ExaTN::Numerics: Space Basis
REVISION: 2019/02/10

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "spaces.hpp"

#include <iostream>

namespace exatn{

namespace numerics{

VectorSpace::VectorSpace(DimExtent space_dim):
 basis_(space_dim), space_name_(""), id_(SOME_SPACE)
{
}

VectorSpace::VectorSpace(DimExtent space_dim, const std::string & space_name):
 basis_(space_dim), space_name_(space_name), id_(SOME_SPACE)
{
}

void VectorSpace::printIt() const
{
 if(space_name_.length() > 0){
  std::cout << "Space {Dimension = " << basis_.getBasisDimension() << "; id = " << id_ << "; Name = " << space_name_ << "}";
 }else{
  std::cout << "Space {Dimension = " << basis_.getBasisDimension() << "; id = " << id_ << "; Name = NONE}";
 }
 return;
}

DimExtent VectorSpace::getSpaceDimension() const
{
 return basis_.getBasisDimension();
}

const std::string & VectorSpace::getSpaceName() const
{
 return space_name_;
}

SpaceId VectorSpace::getSpaceId() const
{
 return id_;
}

void VectorSpace::resetRegisteredId(SpaceId id)
{
 id_=id;
 return;
}

} //namespace numerics

} //namespace exatn
