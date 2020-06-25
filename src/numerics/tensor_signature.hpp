/** ExaTN::Numerics: Tensor signature
REVISION: 2020/06/25

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Tensor signature is an ordered set of tensor dimension specifiers,
     that is, specifiers of the subspaces tensor dimensions are spanned over;
 (b) Registered signature: Tensor dimension specifier consists of a Space Id
     and a Subspace Id, thus associating the tensor dimension with a specific
     registered subspace of a specific registered vector space.
 (c) Anonymous signature: Tensor dimension specifier consists of
     the Space Id = SOME_SPACE, while the Subspace Id specifies
     the offset (first basis vector) in SOME_SPACE.
**/

#ifndef EXATN_NUMERICS_TENSOR_SIGNATURE_HPP_
#define EXATN_NUMERICS_TENSOR_SIGNATURE_HPP_

#include "tensor_basic.hpp"
#include "packable.hpp"
#include "spaces.hpp"

#include <utility>
#include <initializer_list>
#include <vector>

#include <iostream>
#include <fstream>

namespace exatn{

namespace numerics{

class TensorSignature: public Packable {
public:

 /** Create an empty tensor signature. **/
 TensorSignature();

 /** Create a tensor signature by specifying pairs <space_id,subspace_id> for each tensor dimension:
     Case 1: space_id = SOME_SPACE: Then subspace_id refers to the base offset [0..*) in an anonymous abstract space;
     Case 2: space_id != SOME_SPACE: Then space is registered and subspace_id refers to its registered subspace. **/
 TensorSignature(std::initializer_list<std::pair<SpaceId,SubspaceId>> subspaces);
 TensorSignature(const std::vector<std::pair<SpaceId,SubspaceId>> & subspaces);

 /** Create a default tensor signature of std::pair<SOME_SPACE,0> by providing the tensor rank only. **/
 TensorSignature(unsigned int rank);

 /** Create a tensor signature by permuting another tensor signature. **/
 TensorSignature(const TensorSignature & another,          //in: another tensor signature
                 const std::vector<unsigned int> & order); //in: new order (N2O)

 TensorSignature(const TensorSignature &) = default;
 TensorSignature & operator=(const TensorSignature &) = default;
 TensorSignature(TensorSignature &&) noexcept = default;
 TensorSignature & operator=(TensorSignature &&) noexcept = default;
 virtual ~TensorSignature() = default;

 virtual void pack(BytePacket & byte_packet) const override;
 virtual void unpack(BytePacket & byte_packet) override;

 /** Print. **/
 void printIt() const;
 void printItFile(std::ofstream & output_file) const;

 /** Get tensor rank (number of dimensions). **/
 unsigned int getRank() const;

 /** Get the space/subspace id for a specific tensor dimension. **/
 SpaceId getDimSpaceId(unsigned int dim_id) const;
 SubspaceId getDimSubspaceId(unsigned int dim_id) const;
 std::pair<SpaceId,SubspaceId> getDimSpaceAttr(unsigned int dim_id) const;

 /** Get the attributes of all tensor dimensions. **/
 const std::vector<std::pair<SpaceId,SubspaceId>> & getDimSpaceAttrs() const;

 /** Returns TRUE if the tensor signature coincides with another tensor signature. **/
 bool isCongruentTo(const TensorSignature & another) const;

 /** Resets a specific subspace. **/
 void resetDimension(unsigned int dim_id, std::pair<SpaceId,SubspaceId> subspace);

 /** Deletes a specific subspace, reducing the signature rank by one. **/
 void deleteDimension(unsigned int dim_id);

 /** Appends a new subspace at the end, increasing the signature rank by one. **/
 void appendDimension(std::pair<SpaceId,SubspaceId> subspace = {SOME_SPACE,0});

private:

 std::vector<std::pair<SpaceId,SubspaceId>> subspaces_; //tensor signature
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_SIGNATURE_HPP_
