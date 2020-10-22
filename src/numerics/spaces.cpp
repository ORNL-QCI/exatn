/** ExaTN::Numerics: Spaces/Subspaces
REVISION: 2019/07/07

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

VectorSpace::VectorSpace(DimExtent space_dim,
                         const std::string & space_name):
 basis_(space_dim), space_name_(space_name), id_(SOME_SPACE)
{
}

VectorSpace::VectorSpace(DimExtent space_dim,
                         const std::string & space_name,
                         const std::vector<SymmetryRange> & symmetry_subranges):
 basis_(space_dim,symmetry_subranges), space_name_(space_name), id_(SOME_SPACE)
{
}

void VectorSpace::printIt() const
{
 if(space_name_.length() > 0){
  std::cout << "VectorSpace{Dim = " << this->getDimension() << "; id = " << id_ << "; Name = " << space_name_ << "}";
 }else{
  std::cout << "VectorSpace{Dim = " << this->getDimension() << "; id = " << id_ << "; Name = NONE}";
 }
 return;
}

DimExtent VectorSpace::getDimension() const
{
 return basis_.getDimension();
}

const std::string & VectorSpace::getName() const
{
 return space_name_;
}

const std::vector<SymmetryRange> & VectorSpace::getSymmetrySubranges() const
{
 return basis_.getSymmetrySubranges();
}

void VectorSpace::registerSymmetrySubrange(const SymmetryRange subrange)
{
 return basis_.registerSymmetrySubrange(subrange);
}

SpaceId VectorSpace::getRegisteredId() const
{
 return id_;
}

void VectorSpace::resetRegisteredId(SpaceId id)
{
 id_ = id;
 return;
}


Subspace::Subspace(const VectorSpace * vector_space,
                   DimOffset lower_bound,
                   DimOffset upper_bound):
vector_space_(vector_space), lower_bound_(lower_bound), upper_bound_(upper_bound),
subspace_name_(""), id_(UNREG_SUBSPACE)
{
 assert(lower_bound_ <= upper_bound_ && upper_bound_ < vector_space_->getDimension());
}

Subspace::Subspace(const VectorSpace * vector_space,
                   std::pair<DimOffset,DimOffset> bounds):
Subspace(vector_space,std::get<0>(bounds),std::get<1>(bounds))
{
}

Subspace::Subspace(const VectorSpace * vector_space,
                   DimOffset lower_bound,
                   DimOffset upper_bound,
                   const std::string & subspace_name):
vector_space_(vector_space), lower_bound_(lower_bound), upper_bound_(upper_bound),
subspace_name_(subspace_name), id_(UNREG_SUBSPACE)
{
 assert(lower_bound_ <= upper_bound_ && upper_bound_ < vector_space_->getDimension());
}

Subspace::Subspace(const VectorSpace * vector_space,
                   std::pair<DimOffset,DimOffset> bounds,
                   const std::string & subspace_name):
Subspace(vector_space,std::get<0>(bounds),std::get<1>(bounds),subspace_name)
{
}

void Subspace::printIt() const
{
 if(subspace_name_.length() > 0){
  std::cout << "Subspace{Space = " << vector_space_->getName() <<
               "; Lbound = " << lower_bound_ << "; Ubound = " << upper_bound_ <<
               "; id = " << id_ << "; Name = " << subspace_name_ << "}";
 }else{
  std::cout << "Subspace{Space = " << vector_space_->getName() <<
               "; Lbound = " << lower_bound_ << "; Ubound = " << upper_bound_ <<
               "; id = " << id_ << "; Name = NONE}";
 }
 return;
}

DimExtent Subspace::getDimension() const
{
 return upper_bound_ - lower_bound_ + 1;
}

DimOffset Subspace::getLowerBound() const
{
 return lower_bound_;
}

DimOffset Subspace::getUpperBound() const
{
 return upper_bound_;
}

std::pair<DimOffset,DimOffset> Subspace::getBounds() const
{
 return std::make_pair(lower_bound_,upper_bound_);
}

const VectorSpace * Subspace::getVectorSpace() const
{
 return vector_space_;
}

const std::string & Subspace::getName() const
{
 return subspace_name_;
}

SubspaceId Subspace::getRegisteredId() const
{
 return id_;
}

void Subspace::resetRegisteredId(SubspaceId id)
{
 id_ = id;
 return;
}

} //namespace numerics

} //namespace exatn
