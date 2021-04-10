/** ExaTN::Numerics: Tensor network builder: Tree: Tree Tensor Network
REVISION: 2021/04/10

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "network_builder_tree.hpp"
#include "tensor_network.hpp"

#include <initializer_list>
#include <vector>

namespace exatn{

namespace numerics{

NetworkBuilderTree::NetworkBuilderTree():
 max_bond_dim_(1), arity_(2)
{
}


bool NetworkBuilderTree::getParameter(const std::string & name, long long * value) const
{
 bool found = true;
 if(name == "max_bond_dim"){
  *value = max_bond_dim_;
 }else if(name == "arity"){
  *value = arity_;
 }else{
  found = false;
 }
 return found;
}


bool NetworkBuilderTree::setParameter(const std::string & name, long long value)
{
 bool found = true;
 if(name == "max_bond_dim"){
  max_bond_dim_ = value;
 }else if(name == "arity"){
  arity_ = value;
 }else{
  found = false;
 }
 return found;
}


void NetworkBuilderTree::build(TensorNetwork & network)
{
 //Inspect the output tensor:
 auto output_tensor = network.getTensor(0);
 const auto output_tensor_rank = output_tensor->getRank();
 const auto & output_dim_extents = output_tensor->getDimExtents();
 //Build tensor tree by layers:
 std::vector<DimExtent> extents(output_dim_extents);
 unsigned int num_dims = extents.size();
 assert(num_dims > 0);
 unsigned int tensor_id_base = 1, layer = 0;
 //Loop over layers:
 bool not_done = true;
 while(not_done){
  unsigned int num_tensors_in_layer = (num_dims - 1) / arity_ + 1;
  //Loop over tensors within a layer:
  unsigned int num_dims_new = 0;
  for(unsigned int extent_id = 0; extent_id < num_dims; extent_id += arity_){
   unsigned int tens_rank = std::min(static_cast<unsigned int>(arity_),(num_dims - extent_id));
   unsigned int end_decr = 0;
   if(num_dims > arity_){++tens_rank; ++end_decr;} //not root: additional output leg is needed
   //Configure tensor dimension extents:
   DimExtent out_dim_ext = 1;
   std::vector<DimExtent> tens_dims(tens_rank);
   for(unsigned int i = 0; i < (tens_rank - end_decr); ++i){ //input legs
    tens_dims[i] = extents[extent_id + i];
    out_dim_ext *= tens_dims[i];
   }
   if(end_decr){ //output leg
    tens_dims[tens_rank - 1] = std::min(static_cast<DimExtent>(max_bond_dim_),out_dim_ext);
    extents[num_dims_new] = tens_dims[tens_rank - 1];
   }
   //Configure tensor connectivity:
   std::vector<TensorLeg> tens_legs(tens_rank);
   if(layer == 0){
    for(unsigned int i = 0; i < (tens_rank - end_decr); ++i) tens_legs[i] = TensorLeg(0,(extent_id + i));
    if(end_decr){
     tens_legs[tens_rank - 1] = TensorLeg(tensor_id_base + num_tensors_in_layer + (num_dims_new / arity_),
                                          num_dims_new % arity_);
    }
   }else{
    for(unsigned int i = 0; i < (tens_rank - end_decr); ++i){
     unsigned int below_tensor_id = (tensor_id_base - num_dims + extent_id + i);
     tens_legs[i] = TensorLeg(below_tensor_id,(network.getTensor(below_tensor_id)->getRank() - 1));
    }
    if(end_decr){
     tens_legs[tens_rank - 1] = TensorLeg(tensor_id_base + num_tensors_in_layer + (num_dims_new / arity_),
                                          num_dims_new % arity_);
    }
   }
   bool appended = network.placeTensor(tensor_id_base + num_dims_new, //tensor id
                                       std::make_shared<Tensor>("_T"+std::to_string(tensor_id_base + num_dims_new), //tensor name
                                                                tens_dims),
                                       tens_legs,
                                       false,false);
   assert(appended);
   network.getTensor(tensor_id_base + num_dims_new)->rename();
   ++num_dims_new; //next tensor within the layer (each tensor supplies one new dimension)
  }
  tensor_id_base += num_tensors_in_layer;
  not_done = (num_dims_new > 1);
  if(not_done){
   num_dims = num_dims_new;
   ++layer;
  }
 }
 return;
}


std::unique_ptr<NetworkBuilder> NetworkBuilderTree::createNew()
{
 return std::unique_ptr<NetworkBuilder>(new NetworkBuilderTree());
}

} //namespace numerics

} //namespace exatn
