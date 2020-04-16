/** ExaTN::Numerics: Tensor network builder: MPS: Matrix Product State
REVISION: 2020/04/16

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "network_builder_mps.hpp"
#include "tensor_network.hpp"

#include <initializer_list>

namespace exatn{

namespace numerics{

NetworkBuilderMPS::NetworkBuilderMPS():
 max_bond_dim_(1)
{
}


bool NetworkBuilderMPS::getParameter(const std::string & name, long long * value) const
{
 bool found = true;
 if(name == "max_bond_dim"){
  *value = max_bond_dim_;
 }else{
  found = false;
 }
 return found;
}


bool NetworkBuilderMPS::setParameter(const std::string & name, long long value)
{
 bool found = true;
 if(name == "max_bond_dim"){
  max_bond_dim_ = value;
 }else{
  found = false;
 }
 return found;
}


void NetworkBuilderMPS::build(TensorNetwork & network)
{
 bool appended = true;
 //Inspect the output tensor:
 auto output_tensor = network.getTensor(0);
 const auto output_tensor_rank = output_tensor->getRank();
 const auto & output_dim_extents = output_tensor->getDimExtents();
 if(output_tensor_rank == 1){
  appended = network.placeTensor(1, //tensor id
                                 std::make_shared<Tensor>("T"+std::to_string(1), //tensor name
                                  std::initializer_list<DimExtent>{output_dim_extents[0]}),
                                 {TensorLeg{0,0}},
                                 false,
                                 false
                                );
  assert(appended);
 }else if(output_tensor_rank == 2){
  DimExtent bond_dim = std::min(output_dim_extents[0],output_dim_extents[1]);
  appended = network.placeTensor(1, //tensor id
                                 std::make_shared<Tensor>("T"+std::to_string(1), //tensor name
                                  std::initializer_list<DimExtent>{output_dim_extents[0],bond_dim}),
                                 {TensorLeg{0,0},TensorLeg{2,0}},
                                 false,
                                 false
                                );
  assert(appended);
  appended = network.placeTensor(2, //tensor id
                                 std::make_shared<Tensor>("T"+std::to_string(2), //tensor name
                                  std::initializer_list<DimExtent>{bond_dim,output_dim_extents[1]}),
                                 {TensorLeg{1,1},TensorLeg{0,1}},
                                 false,
                                 false
                                );
  assert(appended);
 }else{ //output_tensor_rank > 2
  //Compute internal bond dimensions:
  DimExtent left_bonds[output_tensor_rank], right_bonds[output_tensor_rank];
  DimExtent left_dim = 1;
  for(int i = 0; i < output_tensor_rank; ++i){
   left_bonds[i] = left_dim;
   left_dim *= output_dim_extents[i];
   if(left_dim > max_bond_dim_) left_dim = max_bond_dim_;
  }
  DimExtent right_dim = 1;
  for(int i = (output_tensor_rank - 1); i >= 0; --i){
   right_bonds[i] = right_dim;
   right_dim *= output_dim_extents[i];
   if(right_dim > max_bond_dim_) right_dim = max_bond_dim_;
  }
  //Append left boundary input tensor:
  right_dim = std::min(left_bonds[1],right_bonds[0]);
  appended = network.placeTensor(1, //tensor id
                                 std::make_shared<Tensor>("T"+std::to_string(1), //tensor name
                                  std::initializer_list<DimExtent>{output_dim_extents[0],right_dim}),
                                 {TensorLeg{0,0},TensorLeg{2,0}},
                                 false,
                                 false
                                );
  assert(appended);
  //Append right boundary input tensor:
  left_dim = std::min(right_bonds[output_tensor_rank-2],left_bonds[output_tensor_rank-1]);
  appended = network.placeTensor(output_tensor_rank, //tensor id
                                 std::make_shared<Tensor>("T"+std::to_string(output_tensor_rank), //tensor name
                                  std::initializer_list<DimExtent>{left_dim,output_dim_extents[output_tensor_rank-1]}),
                                 {TensorLeg{output_tensor_rank-1,2},TensorLeg{0,output_tensor_rank-1}},
                                 false,
                                 false
                                );
  assert(appended);
  //Append the internal input tensors:
  for(unsigned int i = 1; i < (output_tensor_rank - 1); ++i){
   left_dim = std::min(left_bonds[i],right_bonds[i-1]);
   right_dim = std::min(right_bonds[i],left_bonds[i+1]);
   if(i == 1){
    appended = network.placeTensor(1+i, //tensor id
                                   std::make_shared<Tensor>("T"+std::to_string(1+i), //tensor name
                                    std::initializer_list<DimExtent>{left_dim,output_dim_extents[i],right_dim}),
                                   {TensorLeg{i,1},TensorLeg{0,i},TensorLeg{i+2,0}},
                                   false,
                                   false
                                  );
   }else{
    appended = network.placeTensor(1+i, //tensor id
                                   std::make_shared<Tensor>("T"+std::to_string(1+i), //tensor name
                                    std::initializer_list<DimExtent>{left_dim,output_dim_extents[i],right_dim}),
                                   {TensorLeg{i,2},TensorLeg{0,i},TensorLeg{i+2,0}},
                                   false,
                                   false
                                  );
   }
   assert(appended);
  }
 }
 return;
}


std::unique_ptr<NetworkBuilder> NetworkBuilderMPS::createNew()
{
 return std::unique_ptr<NetworkBuilder>(new NetworkBuilderMPS());
}

} //namespace numerics

} //namespace exatn
