/** ExaTN::Numerics: Space Basis
REVISION: 2019/03/18

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Space basis is a collection of linear-independent basis vectors.
 (b) An abstract space basis can be further specialized by storing
     specialized basis vectors with additional attributes. By default
     an abstract space basis is only characterized by its dimension.
 (c) Space basis may additionally include symmetry subranges, that is,
     contiguous ranges of basis vectors assigned a specific symmetry id.
**/

#ifndef SPACE_BASIS_HPP_
#define SPACE_BASIS_HPP_

#include "tensor_basic.hpp"
#include "basis_vector.hpp"

#include <vector>

namespace exatn{

namespace numerics{

/** Symmetry subrange: Contiguous range of basis
vectors assigned a specific symmetry id. **/
struct SymmetryRange{
 DimOffset lower;    //lower bound of the subrange
 DimOffset upper;    //upper bound of the subrange
 SymmetryId symm_id; //symmetry id of the subrange
};


/** Vector space basis. **/
class SpaceBasis{
public:

 SpaceBasis(DimExtent space_dim);
 SpaceBasis(DimExtent space_dim,
            const std::vector<SymmetryRange> & symmetry_subranges);

 SpaceBasis(const SpaceBasis & space_basis) = default;
 SpaceBasis & operator=(const SpaceBasis & space_basis) = default;
 SpaceBasis(SpaceBasis && space_basis) = default;
 SpaceBasis & operator=(SpaceBasis && space_basis) = default;
 virtual ~SpaceBasis() = default;

 /** Prints. **/
 void printIt() const;

 /** Returns the basis dimension. **/
 DimExtent getDimension() const;

 /** Returns currently registered symmetry subranges. **/
 const std::vector<SymmetryRange> & getSymmetrySubranges() const;

 /** Registers a symmetry subrange within the basis: A contiguous
 range of basis vectors assigned a specific symmetry id. **/
 void registerSymmetrySubrange(const SymmetryRange subrange);

private:

 DimExtent basis_dim_;                        //basis dimension (number of basis vectors)
 std::vector<SymmetryRange> symmetry_ranges_; //symmetry subranges (may be none)
};

} //namespace numerics

} //namespace exatn

#endif //SPACES_BASIS_HPP_
