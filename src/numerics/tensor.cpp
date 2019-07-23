/** ExaTN::Numerics: Tensor
REVISION: 2019/07/22

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor.hpp"

#include <iostream>
#include <assert.h>

namespace exatn{

namespace numerics{

Tensor::Tensor(const std::string & name,
               const TensorShape & shape,
               const TensorSignature & signature):
name_(name), shape_(shape), signature_(signature)
{
 //DEBUG:
 if(signature_.getRank() != shape_.getRank()) std::cout << "ERROR(Tensor::Tensor): Signature/Shape size mismatch!" << std::endl;
 assert(signature_.getRank() == shape_.getRank());
}

Tensor::Tensor(const std::string & name,
               const TensorShape & shape):
name_(name), shape_(shape), signature_(shape.getRank())
{
}

Tensor::Tensor(const std::string & name):
name_(name)
{
}

Tensor::Tensor(const std::string & name,                    //tensor name
               const Tensor & left_tensor,                  //left tensor
               const Tensor & right_tensor,                 //right tensor
               const std::vector<TensorLeg> & contraction): //tensor contraction pattern
name_(name)
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
  assert(max_out_dim < out_mode);
  //Form the output tensor shape/signature:
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

void Tensor::printIt() const
{
 std::cout << name_; signature_.printIt(); shape_.printIt();
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

TensorHashType Tensor::getTensorHash() const
{
 return reinterpret_cast<TensorHashType>(this);
}

} //namespace numerics

} //namespace exatn
