/** ExaTN::Numerics: Spaces/Subspaces
REVISION: 2021/03/05

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) An abstract vector space is defined by its dimension, N, making it
     a linear span of its N abstract basis vectors. Additonally, symmetry
     subranges can be defined within the space basis, that is, contiguous
     subranges of basis vectors can be assigned a specific symmetry id.
 (b) A specialized vector space is a span of linear-independent
     specialized basis vectors (specialized basis).
 (c) A subspace of a vector space is defined by its encompassing vector space
     and a range of basis vectors it is spanned over.
**/

#ifndef EXATN_NUMERICS_SPACES_HPP_
#define EXATN_NUMERICS_SPACES_HPP_

#include "tensor_basic.hpp"
#include "space_basis.hpp"

#include <utility>
#include <string>
#include <vector>
#include <memory>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class VectorSpace{
public:

 /** Abstract anonymous vector space of a given dimension. **/
 VectorSpace(DimExtent space_dim);
 /** Abstract named vector space of a given dimension. **/
 VectorSpace(DimExtent space_dim,
             const std::string & space_name);
 /** Abstract named vector space of a given dimension with symmetry subranges. **/
 VectorSpace(DimExtent space_dim,
             const std::string & space_name,
             const std::vector<SymmetryRange> & symmetry_subranges);

 VectorSpace(const VectorSpace & vector_space) = default;
 VectorSpace & operator=(const VectorSpace & vector_space) = default;
 VectorSpace(VectorSpace && vector_space) noexcept = default;
 VectorSpace & operator=(VectorSpace && vector_space) noexcept = default;
 virtual ~VectorSpace() = default;

 /** Prints. **/
 void printIt() const;

 /** Returns the space dimension. **/
 DimExtent getDimension() const;

 /** Returns the name of the space. **/
 const std::string & getName() const;

 /** Returns currently defined symmetry subranges. **/
 const std::vector<SymmetryRange> & getSymmetrySubranges() const;

 /** Registers a symmetry subrange within the space:
     A contiguous range of basis vectors assigned a specific symmetry id. **/
 void registerSymmetrySubrange(const SymmetryRange subrange);

 /** Returns the registered space id. **/
 SpaceId getRegisteredId() const;

 friend class SpaceRegister;

private:

 /** Resets the registered space id. **/
 void resetRegisteredId(SpaceId id);

 SpaceBasis basis_;       //basis defining the vector space
 std::string space_name_; //optional space name
 SpaceId id_;             //registered space id (defaults to SOME_SPACE)
};


class Subspace{
public:

 /** Anonymous subspace of a vector space defined by a subrange of basis vectors. **/
 Subspace(const VectorSpace * vector_space,
          DimOffset lower_bound,
          DimOffset upper_bound);
 Subspace(const VectorSpace * vector_space,
          std::pair<DimOffset,DimOffset> bounds);
 /** Named subspace of a vector space defined by a subrange of basis vectors. **/
 Subspace(const VectorSpace * vector_space,
          DimOffset lower_bound,
          DimOffset upper_bound,
          const std::string & subspace_name);
 Subspace(const VectorSpace * vector_space,
          std::pair<DimOffset,DimOffset> bounds,
          const std::string & subspace_name);

 Subspace(const Subspace & subspace) = default;
 Subspace & operator=(const Subspace & subspace) = default;
 Subspace(Subspace && subspace) noexcept = default;
 Subspace & operator=(Subspace && subspace) noexcept = default;
 virtual ~Subspace() = default;

 /** Prints. **/
 void printIt() const;

 /** Returns the subspace dimension. **/
 DimExtent getDimension() const;

 /** Returns the lower bound of the subspace. **/
 DimOffset getLowerBound() const;

 /** Returns the upper bound of the subspace. **/
 DimOffset getUpperBound() const;

 /** Returns the bounds of the subspace. **/
 std::pair<DimOffset,DimOffset> getBounds() const;

 /** Returns a pointer to the vector space the subspace is defined in. **/
 const VectorSpace * getVectorSpace() const;

 /** Returns the name of the subspace. **/
 const std::string & getName() const;

 /** Returns the registered subspace id. **/
 SubspaceId getRegisteredId() const;

 /** Splits the subspace into a given number of smaller subspaces maximally uniformly.
     If the extent of the parental subspace is smaller than the requested number of
     segments, a vector of null pointers will be returned. The produced child subspaces
     are named as {"_" + ParentSubspaceName + "_" + SegmentNumber}. **/
 std::vector<std::shared_ptr<Subspace>> splitUniform(DimExtent num_segments) const;

 friend class SubspaceRegister;

private:

 /** Resets the registered subspace id. **/
 void resetRegisteredId(SubspaceId id);

 const VectorSpace * vector_space_; //non-owning pointer to the vector space
 DimOffset lower_bound_;            //lower bound defining the subspace of the vector space
 DimOffset upper_bound_;            //upper bound defining the subspace of the vector space
 std::string subspace_name_;        //optional subspace name
 SubspaceId id_;                    //registered subspace id (defaults to UNREG_SUBSPACE)
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_SPACES_HPP_
