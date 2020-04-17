/** ExaTN::Numerics: Tensor
REVISION: 2020/04/16

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor.hpp"
#include "tensor_symbol.hpp"

#include <iostream>
#include <cassert>

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

Tensor::Tensor(const std::string & name,
               const TensorShape & shape):
name_(name), shape_(shape), signature_(shape.getRank()), element_type_(TensorElementType::VOID)
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
  unsigned int contr[left_rank+right_rank][2] = {0};
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
 assert(left_tensor_type == right_tensor_type);
 this->setElementType(left_tensor_type);
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

void Tensor::printIt(bool with_hash) const
{
 if(!with_hash){
  std::cout << name_;
 }else{
  std::cout << name_ << "#" << this->getTensorHash();
 }
 signature_.printIt();
 shape_.printIt();
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
 return;
}

void Tensor::rename(const std::string & name)
{
 name_ = name;
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

std::shared_ptr<Tensor> Tensor::createSubtensor(const std::string & name,
                                                const std::vector<int> & mode_mask, int mask_val)
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

const std::list<std::vector<unsigned int>> & Tensor::retrieveIsometries() const
{
 return isometries_;
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
