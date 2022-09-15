/** ExaTN::Numerics: Tensor network builder: Tree: Tree Tensor Network
REVISION: 2022/09/14

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corporation

SPDX-License-Identifier: BSD-3-Clause **/

#include "network_builder_ttn.hpp"
#include "tensor_network.hpp"

#include <initializer_list>
#include <vector>
#include <algorithm>

namespace exatn{

namespace numerics{

NetworkBuilderTTN::NetworkBuilderTTN():
 max_bond_dim_(1), arity_(2), num_states_(1),
 isometric_(0), free_root_(0), add_terminal_(0)
{
}


bool NetworkBuilderTTN::getParameter(const std::string & name, long long * value) const
{
 bool found = true;
 if(name == "max_bond_dim"){
  *value = max_bond_dim_;
 }else if(name == "arity"){
  *value = arity_;
 }else if(name == "num_states"){
  *value = num_states_;
 }else if(name == "isometric"){
  *value = isometric_;
 }else if(name == "free_root"){
  *value = free_root_;
 }else if(name == "add_terminal"){
  *value = add_terminal_;
 }else{
  found = false;
 }
 return found;
}


bool NetworkBuilderTTN::setParameter(const std::string & name, long long value)
{
 bool found = true;
 if(name == "max_bond_dim"){
  max_bond_dim_ = value;
 }else if(name == "arity"){
  arity_ = value;
 }else if(name == "num_states"){
  num_states_ = value;
 }else if(name == "isometric"){
  isometric_ = value;
 }else if(name == "free_root"){
  free_root_ = value;
 }else if(name == "add_terminal"){
  add_terminal_ = value;
 }else{
  found = false;
 }
 return found;
}


void NetworkBuilderTTN::build(TensorNetwork & network, bool tensor_operator)
{
 make_sure(num_states_ <= max_bond_dim_,
  "#ERROR(NetworkBuilderTTN::build): Number of states must not exceed the max bond dimension!");
 //Inspect the output tensor:
 auto output_tensor = network.getTensor(0);
 auto output_tensor_rank = output_tensor->getRank();
 assert(output_tensor_rank > 0);
 const auto & output_dim_extents = output_tensor->getDimExtents();
 if(tensor_operator){
  assert(output_tensor_rank % 2 == 0); //tensor operators are assumed to be of even rank here
  output_tensor_rank /= 2;
  for(unsigned int i = 0; i < output_tensor_rank; ++i){
   assert(output_dim_extents[i] == output_dim_extents[output_tensor_rank+i]);
  }
 }
 //Build tensor tree by layers:
 std::vector<DimExtent> extents(output_tensor_rank);
 std::copy(output_dim_extents.cbegin(),output_dim_extents.cbegin()+output_tensor_rank,extents.begin());
 unsigned int num_dims = extents.size(); assert(num_dims > 0);
 unsigned int tensor_id_base = 1, tree_root_id = 0, tree_root_dim = 0, layer = 0;
 //Loop over layers:
 bool not_done = true;
 while(not_done){
  unsigned int num_tensors_in_layer = (num_dims - 1) / arity_ + 1;
  //Loop over tensors within a layer:
  unsigned int num_dims_new = 0;
  for(unsigned int extent_id = 0; extent_id < num_dims; extent_id += arity_){
   unsigned int tens_rank = std::min(static_cast<unsigned int>(arity_),(num_dims - extent_id));
   unsigned int end_decr = 0;
   if(num_dims > arity_ || num_states_ > 1){++tens_rank; ++end_decr;} //append output leg
   //Configure tensor dimension extents:
   DimExtent out_dim_ext = 1;
   std::vector<DimExtent> tens_dims(tens_rank);
   for(unsigned int i = 0; i < (tens_rank - end_decr); ++i){ //input legs
    tens_dims[i] = extents[extent_id + i];
    out_dim_ext *= tens_dims[i];
   }
   if(end_decr){ //output leg
    if(num_dims > arity_){ //not root
     tens_dims[tens_rank - 1] = std::min(static_cast<DimExtent>(max_bond_dim_),out_dim_ext);
     extents[num_dims_new] = tens_dims[tens_rank - 1];
    }else{ //root
     tens_dims[tens_rank - 1] = num_states_;
     extents[num_dims_new] = tens_dims[tens_rank - 1];
    }
   }
   //Configure tensor connectivity:
   std::vector<TensorLeg> tens_legs(tens_rank);
   if(layer == 0){
    for(unsigned int i = 0; i < (tens_rank - end_decr); ++i) tens_legs[i] = TensorLeg(0,(extent_id + i));
    if(end_decr){
     if(num_dims > arity_){ //not root
      tens_legs[tens_rank - 1] = TensorLeg(tensor_id_base + num_tensors_in_layer + (num_dims_new / arity_),
                                           num_dims_new % arity_);
     }else{ //root
      tens_legs[tens_rank - 1] = TensorLeg(0,output_dim_extents.size());
     }
    }
   }else{
    for(unsigned int i = 0; i < (tens_rank - end_decr); ++i){
     unsigned int below_tensor_id = (tensor_id_base - num_dims + extent_id + i);
     if(tensor_operator){
      tens_legs[i] = TensorLeg(below_tensor_id,(network.getTensor(below_tensor_id)->getRank() / 2));
     }else{
      tens_legs[i] = TensorLeg(below_tensor_id,(network.getTensor(below_tensor_id)->getRank() - 1));
     }
    }
    if(end_decr){
     if(num_dims > arity_){ //not root
      tens_legs[tens_rank - 1] = TensorLeg(tensor_id_base + num_tensors_in_layer + (num_dims_new / arity_),
                                           num_dims_new % arity_);
     }else{ //root
      tens_legs[tens_rank - 1] = TensorLeg(0,output_dim_extents.size());
     }
    }
   }
   //Emplace the tensor:
   const auto new_tensor_id = tensor_id_base + num_dims_new;
   bool appended = network.placeTensor(new_tensor_id, //tensor id
                                       std::make_shared<Tensor>("_T"+std::to_string(new_tensor_id), //tensor name
                                                                tens_dims),
                                       tens_legs,false,false);
   assert(appended);
   //Append additional legs to tree leafs when operator is needed:
   if(tensor_operator && layer == 0){ //tensor tree operator leaf
    auto * tens_conn = network.getTensorConn(new_tensor_id);
    for(unsigned int i = 0; i < (tens_rank - end_decr); ++i){
     const unsigned int output_dim_id = output_tensor_rank + extent_id + i;
     tens_conn->appendLeg(output_dim_extents[output_dim_id],TensorLeg{0,output_dim_id});
    }
   }
   auto tens = network.getTensor(new_tensor_id);
   tens->rename();
   tree_root_id = new_tensor_id; //assumes the last appended tensor is the root
   tree_root_dim = tens_rank - end_decr;
   //Register isometries (if specified):
   if(isometric_ != 0){
    std::vector<unsigned int> iso_dims(tens->getRank() - end_decr);
    unsigned int k = 0;
    for(unsigned int i = 0; i < (tens_rank - end_decr); ++i) iso_dims[k++] = i;
    if(tensor_operator && layer == 0){
     for(unsigned int i = 0; i < (tens_rank - end_decr); ++i) iso_dims[k++] = tens_rank + i;
    }
    tens->registerIsometry(iso_dims);
   }
   ++num_dims_new; //next tensor within the layer (each tensor supplies one new dimension)
  }
  tensor_id_base += num_tensors_in_layer;
  not_done = (num_dims_new > 1);
  if(not_done){
   num_dims = num_dims_new;
   ++layer;
  }
 }
 //Post-processing:
 if(num_states_ > 1){
  //Append the state leg to the tree root tensor:
  auto * output_tens_conn = network.getTensorConn(0);
  output_tens_conn->appendLeg(DimExtent{num_states_},TensorLeg{tree_root_id,tree_root_dim});
 }else{
  //Unregister isometries in the tree root tensor, if needed:
  auto * root_tens_conn = network.getTensorConn(tree_root_id);
  if(isometric_ != 0 && free_root_ != 0) root_tens_conn->unregisterIsometries();
  //Append an order-1 terminal tensor to the tree root tensor, if needed:
  if(add_terminal_ != 0){
   const auto & root_extents = root_tens_conn->getDimExtents();
   DimExtent comb_dim_ext = 1;
   for(const auto & ext: root_extents){
    comb_dim_ext *= ext;
    if(comb_dim_ext >= max_bond_dim_){
     comb_dim_ext = max_bond_dim_;
     break;
    }
   }
   const auto new_tensor_id = tree_root_id + 1;
   root_tens_conn->appendLeg(comb_dim_ext,TensorLeg{new_tensor_id,0});
   bool appended = network.placeTensor(new_tensor_id, //tensor id
                                       std::make_shared<Tensor>("_T"+std::to_string(new_tensor_id), //tensor name
                                                                std::initializer_list<DimExtent>{comb_dim_ext}),
                                       {TensorLeg{tree_root_id,root_tens_conn->getRank()-1}},false,false);
   assert(appended);
   network.getTensor(new_tensor_id)->rename();
  }
 }
 //std::cout << "#DEBUG(exatn::network_builder_ttn): Network built:\n"; network.printIt(); //debug
 return;
}


std::unique_ptr<NetworkBuilder> NetworkBuilderTTN::createNew()
{
 return std::unique_ptr<NetworkBuilder>(new NetworkBuilderTTN());
}

} //namespace numerics

} //namespace exatn
