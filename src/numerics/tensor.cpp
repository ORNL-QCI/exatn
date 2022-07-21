/** ExaTN::Numerics: Tensor
REVISION: 2022/06/07

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corporation

SPDX-License-Identifier: BSD-3-Clause **/

#include "tensor.hpp"
#include "tensor_symbol.hpp"
#include "space_register.hpp"

#include <iostream>
#include <algorithm>
#include <cstring>

namespace exatn{

namespace numerics{

Tensor::Tensor(const std::string & name,
               const TensorShape & shape,
               const TensorSignature & signature):
name_(name), shape_(shape), signature_(signature), element_type_(TensorElementType::VOID)
{
 //DEBUG:
 if(signature_.getRank() != shape_.getRank()) std::cout << "ERROR(Tensor::Tensor): Signature/Shape size mismatch!" << std::endl;
 assert(signature_.getRank() == shape_.getRank());
}

Tensor::Tensor(const TensorShape & shape,
               const TensorSignature & signature):
 Tensor(tensor_hex_name("",getTensorHash()),shape,signature)
{
}

Tensor::Tensor(const std::string & name,
               const TensorShape & shape):
name_(name), shape_(shape), signature_(shape.getRank()), element_type_(TensorElementType::VOID)
{
}

Tensor::Tensor(const TensorShape & shape):
 Tensor(tensor_hex_name("",getTensorHash()),shape)
{
}

Tensor::Tensor(const std::string & name):
name_(name), element_type_(TensorElementType::VOID)
{
}

Tensor::Tensor(const std::string & name,                    //tensor name
               const Tensor & left_tensor,                  //left tensor
               const Tensor & right_tensor,                 //right tensor
               const std::vector<TensorLeg> & contraction): //tensor contraction pattern
name_(name), element_type_(TensorElementType::VOID)
{
 //Import shape/signature of the input tensors:
 auto left_rank = left_tensor.getRank();
 TensorShape left_shape = left_tensor.getShape();
 TensorSignature left_signa = left_tensor.getSignature();
 auto right_rank = right_tensor.getRank();
 TensorShape right_shape = right_tensor.getShape();
 TensorSignature right_signa = right_tensor.getSignature();
 //Extract the output tensor dimensions:
 if(left_rank + right_rank > 0){
  unsigned int out_mode = 0;
  unsigned int inp_mode = 0;
  unsigned int argt = 1; if(left_rank == 0) argt = 2;
  unsigned int max_out_dim = 0;
  unsigned int contr[left_rank+right_rank][2];
  memset(contr, 0, (left_rank+right_rank)*2*sizeof(unsigned int));

  for(const auto & leg: contraction){
   auto tens_id = leg.getTensorId();
   if(tens_id == 0){ //uncontracted leg of either input tensor
    unsigned int out_dim = leg.getDimensionId(); //output tensor mode id
    if(out_dim > max_out_dim) max_out_dim = out_dim;
    contr[out_dim][0] = argt;     //input tensor argument: {1,2}
    contr[out_dim][1] = inp_mode; //input tensor mode id
    ++out_mode;
   }else{
    assert(tens_id == 1 || tens_id == 2); //checking validity of argument <contraction>
   }
   ++inp_mode;
   if(argt == 1 && inp_mode == left_rank){inp_mode = 0; argt = 2;};
  }
  //Form the output tensor shape/signature:
  if(out_mode > 0){ //output tensor is not a scalar
   assert(max_out_dim < out_mode);
   for(unsigned int i = 0; i <= max_out_dim; ++i){
    inp_mode = contr[i][1];
    if(contr[i][0] == 1){
     shape_.appendDimension(left_tensor.getDimExtent(inp_mode));
     signature_.appendDimension(left_tensor.getDimSpaceAttr(inp_mode));
    }else if(contr[i][0] == 2){
     shape_.appendDimension(right_tensor.getDimExtent(inp_mode));
     signature_.appendDimension(right_tensor.getDimSpaceAttr(inp_mode));
    }else{
     std::cout << "#ERROR(Tensor::Tensor): Invalid function argument: contraction: Missing output tensor mode!" << std::endl;
     assert(false); //missing output tensor dimension
    }
   }
  }
 }
 //Set the tensor element type:
 auto left_tensor_type = left_tensor.getElementType();
 auto right_tensor_type = right_tensor.getElementType();
 if(static_cast<int>(left_tensor_type) <= static_cast<int>(right_tensor_type)){
  this->setElementType(left_tensor_type);
 }else{
  this->setElementType(right_tensor_type);
 }
}

std::vector<TensorLeg> & generatePatternForIsometricSelfContraction(const Tensor & tensor,
                                                                    std::vector<TensorLeg> & contraction,
                                                                    unsigned int iso_group_id)
{
 const auto tens_rank = tensor.getRank();
 contraction.resize(tens_rank*2,TensorLeg());
 if(tens_rank > 0){
  std::vector<unsigned int> marks(tens_rank,0);
  const auto iso_group = tensor.retrieveIsometry(iso_group_id);
  for(const auto & iso_dim: iso_group){
   marks[iso_dim] = 1;
   contraction[iso_dim] = TensorLeg{2,iso_dim};
   contraction[iso_dim+tens_rank] = TensorLeg{1,iso_dim};
  }
  const unsigned int niso_size = tens_rank - iso_group.size();
  if(niso_size > 0){
   unsigned int dest_leg = 0;
   for(unsigned int i = 0; i < tens_rank; ++i){
    if(marks[i] == 0){
     contraction[i] = TensorLeg{0,dest_leg};
     contraction[i+tens_rank] = TensorLeg{0,dest_leg+niso_size};
     ++dest_leg;
    }
   }
  }
 }
 return contraction;
}

Tensor::Tensor(const std::string & name,
               const Tensor & isometric_tensor,
               std::vector<TensorLeg> & contraction,
               unsigned int iso_group_id):
 Tensor(name,isometric_tensor,isometric_tensor,
        generatePatternForIsometricSelfContraction(isometric_tensor,contraction,iso_group_id))
{
}

Tensor::Tensor(BytePacket & byte_packet)
{
 unpack(byte_packet);
}

Tensor::Tensor(const Tensor & another,
               const std::vector<unsigned int> & order):
 name_(another.getName()),
 shape_(another.getShape(),order),
 signature_(another.getSignature(),order),
 element_type_(another.getElementType()),
 isometries_(another.retrieveIsometries())
{
 if(!(isometries_.empty())){
  const auto rank = order.size();
  unsigned int o2n[rank];
  for(unsigned int i = 0; i < rank; ++i) o2n[order[i]] = i;
  for(auto & iso_group: isometries_){
   for(auto & old_dim: iso_group) old_dim = o2n[old_dim];
  }
 }
}

std::shared_ptr<Tensor> Tensor::clone() const
{
 return makeSharedTensor(*this);
}

void Tensor::pack(BytePacket & byte_packet) const
{
 const std::size_t name_len = name_.length();
 appendToBytePacket(&byte_packet,name_len);
 for(std::size_t i = 0; i < name_len; ++i) appendToBytePacket(&byte_packet,name_[i]);
 shape_.pack(byte_packet);
 signature_.pack(byte_packet);
 appendToBytePacket(&byte_packet,element_type_);
 const std::size_t num_isometries = isometries_.size();
 appendToBytePacket(&byte_packet,num_isometries);
 for(const auto & isometry: isometries_){
  const std::size_t num_vertices = isometry.size();
  appendToBytePacket(&byte_packet,num_vertices);
  for(const auto & vertex_id: isometry) appendToBytePacket(&byte_packet,vertex_id);
 }
 return;
}

void Tensor::unpack(BytePacket & byte_packet)
{
 std::size_t name_len = 0;
 extractFromBytePacket(&byte_packet,name_len);
 name_.resize(name_len);
 for(std::size_t i = 0; i < name_len; ++i) extractFromBytePacket(&byte_packet,name_[i]);
 shape_.unpack(byte_packet);
 signature_.unpack(byte_packet);
 extractFromBytePacket(&byte_packet,element_type_);
 isometries_.clear();
 std::size_t num_isometries = 0;
 extractFromBytePacket(&byte_packet,num_isometries);
 isometries_.resize(num_isometries);
 for(auto & isometry: isometries_){
  std::size_t num_vertices = 0;
  extractFromBytePacket(&byte_packet,num_vertices);
  isometry.resize(num_vertices);
  for(auto & vertex_id: isometry) extractFromBytePacket(&byte_packet,vertex_id);
 }
 return;
}

bool Tensor::isComposite() const
{
 return false;
}

bool Tensor::isConformantTo(const Tensor & another) const
{
 bool ans = true;
 if(another.isComposite()){
  ans = another.isConformantTo(*this);
 }else{
  ans = isCongruentTo(another);
 }
 return ans;
}

void Tensor::printIt(bool with_hash) const
{
 if(!with_hash){
  std::cout << name_;
 }else{
  std::cout << name_ << "#" << this->getTensorHash();
 }
 signature_.printIt();
 shape_.printIt();
 for(const auto & iso_group: isometries_){
  std::cout << "[";
  for(auto iso_dim = iso_group.cbegin(); iso_dim != iso_group.cend(); ++iso_dim){
   if(iso_dim != iso_group.cbegin()) std::cout << ",";
   std::cout << (*iso_dim);
  }
  std::cout << "]";
 }
 return;
}

void Tensor::printItFile(std::ofstream & output_file, bool with_hash) const
{
 if(!with_hash){
  output_file << name_;
 }else{
  output_file << name_ << "#" << this->getTensorHash();
 }
 signature_.printItFile(output_file);
 shape_.printItFile(output_file);
 for(const auto & iso_group: isometries_){
  output_file << "[";
  for(auto iso_dim = iso_group.cbegin(); iso_dim != iso_group.cend(); ++iso_dim){
   if(iso_dim != iso_group.cbegin()) output_file << ",";
   output_file << (*iso_dim);
  }
  output_file << "]";
 }
 return;
}

void Tensor::rename(const std::string & name)
{
 name_ = name;
 return;
}

void Tensor::rename()
{
 name_ = tensor_hex_name("",this->getTensorHash());
 return;
}

const std::string & Tensor::getName() const
{
 return name_;
}

unsigned int Tensor::getRank() const
{
 return shape_.getRank();
}

std::size_t Tensor::getVolume() const
{
 return static_cast<std::size_t>(shape_.getVolume());
}

std::size_t Tensor::getSize() const
{
 return static_cast<std::size_t>(shape_.getVolume()) * tensor_element_type_size(element_type_);
}

const TensorShape & Tensor::getShape() const
{
 return shape_;
}

const TensorSignature & Tensor::getSignature() const
{
 return signature_;
}

DimExtent Tensor::getDimExtent(unsigned int dim_id) const
{
 return shape_.getDimExtent(dim_id);
}

const std::vector<DimExtent> & Tensor::getDimExtents() const
{
 return shape_.getDimExtents();
}

const std::vector<DimExtent> Tensor::getDimStrides(DimExtent * volume) const
{
 return shape_.getDimStrides(volume);
}

SpaceId Tensor::getDimSpaceId(unsigned int dim_id) const
{
 return signature_.getDimSpaceId(dim_id);
}

SubspaceId Tensor::getDimSubspaceId(unsigned int dim_id) const
{
 return signature_.getDimSubspaceId(dim_id);
}

std::pair<SpaceId,SubspaceId> Tensor::getDimSpaceAttr(unsigned int dim_id) const
{
 return signature_.getDimSpaceAttr(dim_id);
}

bool Tensor::isCongruentTo(const Tensor & another) const
{
 return shape_.isCongruentTo(another.getShape()) &&
        signature_.isCongruentTo(another.getSignature());
}

void Tensor::deleteDimension(unsigned int dim_id)
{
 signature_.deleteDimension(dim_id);
 shape_.deleteDimension(dim_id);
 return;
}

void Tensor::appendDimension(std::pair<SpaceId,SubspaceId> subspace, DimExtent dim_extent)
{
 signature_.appendDimension(subspace);
 shape_.appendDimension(dim_extent);
 return;
}

void Tensor::appendDimension(DimExtent dim_extent)
{
 this->appendDimension(std::pair<SpaceId,SubspaceId>{SOME_SPACE,0},dim_extent);
 return;
}

void Tensor::replaceDimension(unsigned int dim_id,
                              std::pair<SpaceId,SubspaceId> subspace,
                              DimExtent dim_extent)
{
 this->signature_.resetDimension(dim_id,subspace);
 this->shape_.resetDimension(dim_id,dim_extent);
 return;
}

void Tensor::replaceDimension(unsigned int dim_id,
                              std::pair<SpaceId,SubspaceId> subspace)
{
 this->signature_.resetDimension(dim_id,subspace);
 return;
}

void Tensor::replaceDimension(unsigned int dim_id,
                              DimExtent dim_extent)
{
 this->shape_.resetDimension(dim_id,dim_extent);
 return;
}

std::shared_ptr<Tensor> Tensor::createSubtensor(const std::string & name,
                                                const std::vector<int> & mode_mask, int mask_val) const
{
 const auto tensor_rank = this->getRank();
 assert(tensor_rank == mode_mask.size());
 auto subtensor = std::make_shared<Tensor>(name); assert(subtensor);
 for(unsigned int i = 0; i < tensor_rank; ++i){
  if(mode_mask[i] == mask_val){
   subtensor->appendDimension(this->getDimSpaceAttr(i),this->getDimExtent(i));
  }
 }
 return subtensor;
}

std::shared_ptr<Tensor> Tensor::createSubtensor(const std::vector<SubspaceId> & subspaces,
                                                const std::vector<DimExtent> & dim_extents) const
{
 assert(subspaces.size() == this->getRank());
 assert(dim_extents.size() == this->getRank());
 auto subtensor = std::make_shared<Tensor>(*this);
 const auto tens_rank = subtensor->getRank();
 for(unsigned int i = 0; i < tens_rank; ++i){
  subtensor->replaceDimension(i,std::make_pair(this->getDimSpaceId(i),subspaces[i]),dim_extents[i]);
 }
 return subtensor;
}

std::vector<std::shared_ptr<Tensor>> Tensor::createSubtensors(unsigned int dim_id,
                                                              DimExtent num_segments) const
{
 const auto tensor_rank = getRank();
 assert(tensor_rank > 0);
 assert(dim_id < tensor_rank);
 assert(num_segments <= getDimExtent(dim_id));
 std::vector<std::shared_ptr<Tensor>> subtensors(num_segments);
 std::vector<SubspaceId> subspace_ids(tensor_rank);
 std::vector<DimExtent> dim_extents(tensor_rank);
 for(unsigned int i = 0; i < tensor_rank; ++i) subspace_ids[i] = getDimSubspaceId(i);
 for(unsigned int i = 0; i < tensor_rank; ++i) dim_extents[i] = getDimExtent(i);
 auto space_reg = getSpaceRegister();
 const auto space_attr = getDimSpaceAttr(dim_id);
 if(space_attr.first == SOME_SPACE){
  const auto * some_space = space_reg->getSpace(SOME_SPACE);
  Subspace parental_subspace(some_space,0,getDimExtent(dim_id)-1);
  auto subspaces = parental_subspace.splitUniform(num_segments);
  for(unsigned int i = 0; i < num_segments; ++i){
   subspace_ids[dim_id] = subspaces[i]->getLowerBound();
   dim_extents[dim_id] = subspaces[i]->getDimension();
   subtensors[i] = createSubtensor(subspace_ids,dim_extents);
   subtensors[i]->rename();
  }
 }else{
  const auto * parental_subspace = space_reg->getSubspace(space_attr.first,space_attr.second);
  auto subspaces = parental_subspace->splitUniform(num_segments);
  for(unsigned int i = 0; i < num_segments; ++i){
   subspace_ids[dim_id] = space_reg->registerSubspace(subspaces[i]);
   dim_extents[dim_id] = subspaces[i]->getDimension();
   subtensors[i] = createSubtensor(subspace_ids,dim_extents);
   subtensors[i]->rename();
  }
 }
 return subtensors;
}

void Tensor::setElementType(TensorElementType element_type)
{
 element_type_ = element_type;
}

TensorElementType Tensor::getElementType() const
{
 return element_type_;
}

void Tensor::registerIsometry(const std::vector<unsigned int> & isometry)
{
 const auto tensor_rank = this->getRank();
 assert(isometry.size() <= tensor_rank);
 for(const auto & dim: isometry) assert(dim < tensor_rank);
 if(isometry.size() > 0) isometries_.emplace_back(isometry);
 return;
}

bool Tensor::hasIsometries() const
{
 return !(isometries_.empty());
}

const std::list<std::vector<unsigned int>> & Tensor::retrieveIsometries() const
{
 return isometries_;
}

std::vector<unsigned int> Tensor::retrieveIsometry(unsigned int iso_group_id) const
{
 make_sure(iso_group_id < isometries_.size(),"#ERROR(exatn::Tensor::retrieveIsometry): Invalid isometric group id!");
 std::vector<unsigned int> dims;
 unsigned int iso_id = 0;
 for(const auto & iso_group: isometries_){
  if(iso_id == iso_group_id){
   dims = iso_group;
   break;
  }
  ++iso_id;
 }
 return dims;
}

std::vector<unsigned int> Tensor::retrieveIsometryComplement(unsigned int iso_group_id) const
{
 make_sure(iso_group_id < isometries_.size(),"#ERROR(exatn::Tensor::retrieveIsometryComplement): Invalid isometric group id!");
 std::vector<unsigned int> dims;
 unsigned int iso_id = 0;
 for(const auto & iso_group: isometries_){
  if(iso_id == iso_group_id){
   const auto tens_rank = getRank();
   dims.resize(tens_rank-iso_group.size());
   if(dims.size() > 0){
    std::vector<unsigned int> marks(tens_rank,0);
    for(const auto & iso_dim: iso_group) marks[iso_dim] = 1;
    unsigned int j = 0;
    for(unsigned int i = 0; i < tens_rank; ++i){
     if(marks[i] == 0) dims[j++] = i;
    }
   }
   break;
  }
  ++iso_id;
 }
 return dims;
}

std::vector<unsigned int> Tensor::retrieveNonisometricDimensions() const
{
 const auto tens_rank = getRank();
 std::vector<unsigned int> dims;
 if(tens_rank > 0){
  dims.reserve(tens_rank);
  std::vector<unsigned int> marks(tens_rank,0);
  for(const auto & iso_group: isometries_){
   for(const auto & iso_dim: iso_group) marks[iso_dim] = 1;
  }
  for(unsigned int i = 0; i < tens_rank; ++i){
   if(marks[i] == 0) dims.emplace_back(i);
  }
 }
 return dims;
}

void Tensor::unregisterIsometries()
{
 isometries_.clear();
 return;
}

bool Tensor::withIsometricDimension(unsigned int dim_id, const std::vector<unsigned int> ** iso_group) const
{
 bool found = false;
 for(const auto & group: isometries_){
  if(std::find(group.cbegin(),group.cend(),dim_id) != group.cend()){
   if(iso_group != nullptr) *iso_group = &group;
   found = true;
   break;
  }
 }
 return found;
}

TensorHashType Tensor::getTensorHash() const
{
 return reinterpret_cast<TensorHashType>((void*)this);
}

std::string generateTensorName(const Tensor & tensor,
                               const std::string & prefix)
{
 return tensor_hex_name(prefix,tensor.getTensorHash());
}

} //namespace numerics

} //namespace exatn
