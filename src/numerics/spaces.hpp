/** ExaTN::Numerics: Spaces/Subspaces
REVISION: 2019/02/10

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef SPACES_HPP_
#define SPACES_HPP_

#include "tensor_basic.hpp"
#include "space_basis.hpp"

#include <string>

namespace exatn{

namespace numerics{

class VectorSpace{
public:

 VectorSpace(DimExtent space_dim);
 VectorSpace(DimExtent space_dim, const std::string & space_name);

 VectorSpace(const VectorSpace & vector_space) = default;
 VectorSpace & operator=(const VectorSpace & vector_space) = default;
 VectorSpace(VectorSpace && vector_space) = default;
 VectorSpace & operator=(VectorSpace && vector_space) = default;
 virtual ~VectorSpace() = default;

 /** Prints. **/
 void printIt() const;

 /** Returns the space dimension. **/
 DimExtent getSpaceDimension() const;

 /** Returns the space name. **/
 const std::string & getSpaceName() const;

 /** Returns the registered space id. **/
 SpaceId getSpaceId() const;

 /** Sets a registered space id. **/
 void resetRegisteredId(SpaceId id);

private:

 SpaceBasis basis_;       //basis defining the vector space
 std::string space_name_; //optional space name
 SpaceId id_;             //registered space id

};

} //namespace numerics

} //namespace exatn

#endif //SPACES_HPP_
