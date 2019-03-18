/** ExaTN::Numerics: Spaces/Subspaces
REVISION: 2019/03/18

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

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

#ifndef SPACES_HPP_
#define SPACES_HPP_

#include "tensor_basic.hpp"
#include "space_basis.hpp"

#include <utility>
#include <string>
#include <vector>

namespace exatn{

namespace numerics{

class VectorSpace{
public:

 VectorSpace(DimExtent space_dim);
 VectorSpace(DimExtent space_dim,
             const std::string & space_name);
 VectorSpace(DimExtent space_dim,
             const std::string & space_name,
             const std::vector<SymmetryRange> & symmetry_subranges);

 VectorSpace(const VectorSpace & vector_space) = default;
 VectorSpace & operator=(const VectorSpace & vector_space) = default;
 VectorSpace(VectorSpace && vector_space) = default;
 VectorSpace & operator=(VectorSpace && vector_space) = default;
 virtual ~VectorSpace() = default;

 /** Prints. **/
 void printIt() const;

 /** Returns the space dimension. **/
 DimExtent getDimension() const;

 /** Returns the name of the space. **/
 const std::string & getName() const;

 /** Returns currently defined symmetry subranges. **/
 const std::vector<SymmetryRange> & getSymmetrySubranges() const;

 /** Returns the registered space id. **/
 SpaceId getRegisteredId() const;

 friend class SpaceRegister;

private:

 /** Resets the registered space id. **/
 void resetRegisteredId(SpaceId id);

 SpaceBasis basis_;       //basis defining the vector space
 std::string space_name_; //optional space name
 SpaceId id_;             //registered space id

};


class Subspace{
public:

 Subspace(const VectorSpace * vector_space,
          DimOffset lower_bound,
          DimOffset upper_bound);
 Subspace(const VectorSpace * vector_space,
          DimOffset lower_bound,
          DimOffset upper_bound,
          const std::string & subspace_name);

 Subspace(const Subspace & subspace) = default;
 Subspace & operator=(const Subspace & subspace) = default;
 Subspace(Subspace && subspace) = default;
 Subspace & operator=(Subspace && subspace) = default;
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

 /** Returns a pointer to the vector space the subspace is defined on. **/
 const VectorSpace * getVectorSpace() const;

 /** Returns the name of the subspace. **/
 const std::string & getName() const;

 /** Returns the registered subspace id. **/
 SubspaceId getRegisteredId() const;

 friend class SubspaceRegister;

private:

 /** Resets the registered subspace id. **/
 void resetRegisteredId(SubspaceId id);

 const VectorSpace * vector_space_; //non-owning pointer to the vector space
 DimOffset lower_bound_;            //lower bound defining the subspace of the vector space
 DimOffset upper_bound_;            //upper bound defining the subspace of the vector space
 std::string subspace_name_;        //optional subspace name
 SubspaceId id_;                    //registered subspace id

};

} //namespace numerics

} //namespace exatn

#endif //SPACES_HPP_
