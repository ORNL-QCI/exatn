/** ExaTN::Numerics: Tensor signature
REVISION: 2019/02/12

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef TENSOR_SIGNATURE_HPP_
#define TENSOR_SIGNATURE_HPP_

#include "tensor_basic.hpp"
#include "spaces.hpp"

#include <utility>
#include <initializer_list>
#include <vector>

namespace exatn{

namespace numerics{

class TensorSignature{
public:

 /** Create a tensor signature by specifying pairs <space_id,subspace_id> for each tensor dimension:
     Case 1: space_id = SOME_SPACE: Then subspace_id refers to the base offset [0..*) in some abstract space;
     Case 2: space_id != SOME_SPACE: Then space is registered and subspace_id refers to its registered subspace. **/
 TensorSignature(std::initializer_list<std::pair<SpaceId,SubspaceId>> subspaces);
 TensorSignature(const std::vector<std::pair<SpaceId,SubspaceId>> & subspaces);

 /** Create a default tensor signature of std::pair<SOME_SPACE,0> by providing tensor rank only. **/
 TensorSignature(unsigned int rank);

 TensorSignature(const TensorSignature & tens_signature) = default;
 TensorSignature & operator=(const TensorSignature & tens_signature) = default;
 TensorSignature(TensorSignature && tens_signature) = default;
 TensorSignature & operator=(TensorSignature && tens_signature) = default;
 virtual ~TensorSignature() = default;

 /** Print. **/
 void printIt() const;

 /** Get tensor rank (number of dimensions). **/
 unsigned int getRank() const;

 /** Get the space/subspace id for a specific tensor dimension. **/
 SpaceId getDimSpaceId(unsigned int dim_id) const;
 SubspaceId getDimSubspaceId(unsigned int dim_id) const;
 std::pair<SpaceId,SubspaceId> getDimSpaceAttr(unsigned int dim_id) const;

private:

 std::vector<std::pair<SpaceId,SubspaceId>> subspaces_;
};

} //namespace numerics

} //namespace exatn

#endif //TENSOR_SIGNATURE_HPP_
