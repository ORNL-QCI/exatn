/** ExaTN::Numerics: Tensor network
REVISION: 2019/07/16

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_network.hpp"

#include <iostream>
#include <assert.h>

namespace exatn{

namespace numerics{

TensorNetwork::TensorNetwork():
 explicit_output_(0), finalized_(0)
{
 tensors_.emplace( //output tensor (id = 0)
                  std::make_pair(
                   0U,
                   TensorConn(std::make_shared<Tensor>("_SMOKY_TENSOR_"),0U,std::vector<TensorLeg>())
                  )
                 );
}


TensorNetwork::TensorNetwork(const std::string & name):
 explicit_output_(0), finalized_(0), name_(name)
{
 tensors_.emplace( //output tensor (id = 0)
                  std::make_pair(
                   0U,
                   TensorConn(std::make_shared<Tensor>(name),0U,std::vector<TensorLeg>())
                  )
                 );
}


TensorNetwork::TensorNetwork(const std::string & name,
                             std::shared_ptr<Tensor> output_tensor,
                             const std::vector<TensorLeg> & output_legs):
 explicit_output_(1), finalized_(0), name_(name)
{
 tensors_.emplace( //output tensor (id = 0)
                  std::make_pair(
                   0U,
                   TensorConn(output_tensor,0U,output_legs)
                  )
                 );
}


void TensorNetwork::printIt() const
{
 std::cout << "TensorNetwork[" << name_ << "](" << this->getNumTensors() << "){" << std::endl;
 for(const auto & kv: tensors_) kv.second.printIt();
 std::cout << "}" << std::endl;
 return;
}


bool TensorNetwork::isEmpty() const
{
 return (tensors_.size() <= 1); //only output tensor exists => still empty
}


bool TensorNetwork::isExplicit() const
{
 return (explicit_output_ != 0);
}


bool TensorNetwork::isFinalized() const
{
 return (finalized_ != 0);
}


unsigned int TensorNetwork::getNumTensors() const
{
 return static_cast<unsigned int>(tensors_.size() - 1); //output tensor is not counted
}


const std::string & TensorNetwork::getName() const
{
 return name_;
}


TensorConn * TensorNetwork::getTensorConn(unsigned int tensor_id)
{
 auto it = tensors_.find(tensor_id);
 if(it == tensors_.end()) return nullptr;
 return &(it->second);
}


std::vector<TensorConn*> TensorNetwork::getTensorConnAll()
{
 std::vector<TensorConn*> tensors(this->getNumTensors(),nullptr);
 unsigned int i = 0;
 for(auto & kv: tensors_){
  if(kv.first != 0) tensors[i++] = &(kv.second);
 }
 return tensors;
}


std::shared_ptr<Tensor> TensorNetwork::getTensor(unsigned int tensor_id)
{
 auto it = tensors_.find(tensor_id);
 if(it == tensors_.end()) return std::shared_ptr<Tensor>(nullptr);
 return (it->second).getTensor();
}


bool TensorNetwork::finalize(bool check_validity)
{
 if(finalized_ == 0){
  if(this->isEmpty()){ //empty networks cannot be finalized
   std::cout << "#ERROR(TensorNetwork::finalize): Empty tensor network cannot be finalized!" << std::endl;
   return false;
  }
  finalized_ = 1;
  if(check_validity){
   if(!checkConnections()){
    finalized_ = 0;
    std::cout << "#ERROR(TensorNetwork::finalize): Invalid connectivity prevents tensor network finalization!" << std::endl;
    return false;
   }
  }
 }
 return true;
}


bool TensorNetwork::checkConnections(unsigned int tensor_id)
{
 assert(finalized_ != 0); //tensor network must be in finalized state
 auto * tensor = this->getTensorConn(tensor_id);
 assert(tensor != nullptr); //invalid tensor_id
 auto tensor_rank = tensor->getNumLegs();
 for(unsigned int i = 0; i < tensor_rank; ++i){
  const auto & tensor_leg = tensor->getTensorLeg(i);
  auto other_tensor_id = tensor_leg.getTensorId();
  auto other_tensor_leg_id = tensor_leg.getDimensionId();
  auto * other_tensor = this->getTensorConn(other_tensor_id);
  assert(other_tensor != nullptr); //unable to find the linked tensor
  const auto & other_tensor_leg = other_tensor->getTensorLeg(other_tensor_leg_id);
  if(other_tensor_leg.getTensorId() != tensor_id ||
     other_tensor_leg.getDimensionId() != i ||
     other_tensor_leg.getDirection() != reverseLegDirection(tensor_leg.getDirection())) return false;
 }
 return true;
}


bool TensorNetwork::checkConnections()
{
 assert(finalized_ != 0); //tensor network must be in finalized state
 for(const auto & kv: tensors_){
  if(!checkConnections(kv.first)) return false;
 }
 return true;
}


void TensorNetwork::updateConnections(unsigned int tensor_id)
{
 assert(finalized_ != 0); //tensor network must be in finalized state
 auto * tensor = this->getTensorConn(tensor_id);
 assert(tensor != nullptr); //invalid tensor_id
 auto tensor_rank = tensor->getNumLegs();
 for(unsigned int i = 0; i < tensor_rank; ++i){
  const auto & tensor_leg = tensor->getTensorLeg(i);
  auto other_tensor_id = tensor_leg.getTensorId();
  auto other_tensor_leg_id = tensor_leg.getDimensionId();
  auto * other_tensor = this->getTensorConn(other_tensor_id);
  assert(other_tensor != nullptr); //unable to find the linked tensor
  auto other_tensor_leg = other_tensor->getTensorLeg(other_tensor_leg_id);
  other_tensor_leg.resetTensorId(tensor_id);
  other_tensor_leg.resetDimensionId(i);
  other_tensor->resetLeg(other_tensor_leg_id,other_tensor_leg);
 }
 return;
}


bool TensorNetwork::appendTensor(unsigned int tensor_id,                     //in: tensor id (unique within the tensor network)
                                 std::shared_ptr<Tensor> tensor,             //in: appended tensor
                                 const std::vector<TensorLeg> & connections) //in: tensor connections (fully specified)
{
 if(explicit_output_ == 0){
  std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid request: " <<
   "Appending a tensor via explicit connections to the tensor network that is missing a full output tensor!" << std::endl;
  return false;
 }
 if(finalized_ != 0){
  std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid request: " <<
   "Appending a tensor via explicit connections to the tensor network that has been finalized!" << std::endl;
  return false;
 }
 //Check the validity of new connections:
 unsigned int mode = 0;
 for(const auto & leg: connections){
  const auto * tensconn = this->getTensorConn(leg.getTensorId());
  if(tensconn != nullptr){ //connected tensor is already in the tensor network
   const auto & tens_legs = tensconn->getTensorLegs();
   const auto & tens_leg = tens_legs[leg.getDimensionId()];
   if(tens_leg.getTensorId() != tensor_id || tens_leg.getDimensionId() != mode){
    std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid argument: Connections are invalid!" << std::endl;
    return false;
   }
  }
  ++mode;
 }
 //Append the tensor to the tensor network:
 auto new_pos = tensors_.emplace(std::pair<unsigned int, TensorConn>(
                                  tensor_id,TensorConn(tensor,tensor_id,connections)
                                 )
                                );
 if(!(new_pos.second)){
  std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid request: " <<
   "A tensor with id " << tensor_id << " already exists in the tensor network!" << std::endl;
  return false;
 }
 return true;
}


bool TensorNetwork::appendTensor(unsigned int tensor_id,                                             //in: tensor id (unique within the tensor network)
                                 std::shared_ptr<Tensor> tensor,                                     //in: appended tensor
                                 const std::vector<std::pair<unsigned int, unsigned int>> & pairing, //in: leg pairing: output tensor mode -> appended tensor mode
                                 const std::vector<LegDirection> & leg_dir)                          //in: optional leg direction (for all tensor modes)
{
 if(explicit_output_ != 0){
  std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid request: " <<
   "Appending a tensor via implicit pairing with the output tensor, but the output tensor is explicit!" << std::endl;
  return false;
 }
 //Check validity of leg pairing:
 auto tensor_rank = tensor->getRank();
 bool dir_present = (leg_dir.size() > 0);
 if(dir_present && leg_dir.size() != tensor_rank){
  std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid argument: Incomplete vector of leg directions!" << std::endl;
  return false;
 }
 auto * output_tensor = this->getTensorConn(0);
 assert(output_tensor != nullptr); //output tensor must be present
 auto output_rank = output_tensor->getNumLegs();
 if(output_rank > 0 && tensor_rank > 0){
  int ouf[output_rank] = {0};
  int tef[tensor_rank] = {0};
  for(const auto & link: pairing){
   if(link.first >= output_rank || link.second >= tensor_rank){
    std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid argument: Invalid leg pairing!" << std::endl;
    return false;
   }
   if(ouf[link.first]++ != 0){
    std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid argument: Pairing: Repeated output leg!" << std::endl;
    return false;
   }
   if(tef[link.second]++ != 0){
    std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid argument: Pairing: Repeated new tensor leg!" << std::endl;
    return false;
   }
  }
 }else{
  if(pairing.size() > 0){
   std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid argument: Pairing: Pairing on a scalar tensor!" << std::endl;
   return false;
  }
 }
 //Pair legs of the new tensor with the input tensors of the tensor network:
 if(tensor_rank > 0){ //true tensor
  std::vector<TensorLeg> new_tensor_legs(tensor_rank,TensorLeg(0,0)); //placeholders for legs
  if(pairing.size() > 0){
   std::vector<unsigned int> matched_output_legs(pairing.size(),0);
   unsigned int mode = 0;
   for(const auto & link: pairing){
    const auto & output_tensor_leg_id = link.first;
    const auto & tensor_leg_id = link.second;
    auto output_tensor_leg = output_tensor->getTensorLeg(output_tensor_leg_id);
    const auto input_tensor_id = output_tensor_leg.getTensorId();
    const auto input_tensor_leg_id = output_tensor_leg.getDimensionId();
    auto * input_tensor = this->getTensorConn(input_tensor_id);
    assert(input_tensor != nullptr);
    auto input_tensor_leg = input_tensor->getTensorLeg(input_tensor_leg_id);
    input_tensor_leg.resetTensorId(tensor_id);
    input_tensor_leg.resetDimensionId(tensor_leg_id);
    input_tensor->resetLeg(input_tensor_leg_id,input_tensor_leg);
    new_tensor_legs[tensor_leg_id].resetTensorId(input_tensor_id);
    new_tensor_legs[tensor_leg_id].resetDimensionId(input_tensor_leg_id);
    if(dir_present){
     new_tensor_legs[tensor_leg_id].resetDirection(leg_dir[tensor_leg_id]);
     if(input_tensor_leg.getDirection() != reverseLegDirection(new_tensor_legs[tensor_leg_id].getDirection())){
      std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid argument: Leg directions: Pairing leg direction mismatch!" << std::endl;
      return false;
     }
    }else{
     new_tensor_legs[tensor_leg_id].resetDirection(reverseLegDirection(input_tensor_leg.getDirection()));
    }
    matched_output_legs[mode++] = output_tensor_leg_id;
   }
   //Delete matched legs from the output tensor:
   output_tensor->deleteLegs(matched_output_legs);
   updateConnections(0); //update tensor network connections due to deletion of the matched output tensor legs
  }
  //Append unpaired legs of the new tensor to the output tensor of the network:
  output_rank = output_tensor->getNumLegs();
  unsigned int mode = 0;
  for(auto & leg: new_tensor_legs){
   if(leg.getTensorId() == 0){ //unpaired tensor leg
    LegDirection dir = LegDirection::UNDIRECT;
    if(dir_present) dir = leg_dir[mode];
    leg.resetDimensionId(output_rank);
    leg.resetDirection(dir);
    output_tensor->appendLeg(tensor->getDimSpaceAttr(mode),tensor->getDimExtent(mode),
                             TensorLeg(tensor_id,mode,reverseLegDirection(dir)));
    output_rank = output_tensor->getNumLegs();
   }
   ++mode;
  }
  auto new_pos = tensors_.emplace(std::pair<unsigned int, TensorConn>(
                                   tensor_id,TensorConn(tensor,tensor_id,new_tensor_legs)
                                  )
                                 );
  if(!(new_pos.second)){
   std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid request: " <<
    "A tensor with id " << tensor_id << " already exists in the tensor network!" << std::endl;
   return false;
  }
 }else{ //scalar tensor
  auto new_pos = tensors_.emplace(std::pair<unsigned int, TensorConn>(
                                   tensor_id,TensorConn(tensor,tensor_id,std::vector<TensorLeg>())
                                  )
                                 );
  if(!(new_pos.second)){
   std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid request: " <<
    "A tensor with id " << tensor_id << " already exists in the tensor network!" << std::endl;
   return false;
  }
 }
 finalized_ = 1; //implicit leg pairing always keeps the tensor network in a finalized state
 return true;
}


bool TensorNetwork::appendTensorNetwork(TensorNetwork && network,                                           //in: appended tensor network
                                        const std::vector<std::pair<unsigned int, unsigned int>> & pairing) //in: leg pairing: output tensor mode (primary) -> output tensor mode (appended)
{
 if(!((*this).isFinalized()) || !(network.isFinalized())){
  std::cout << "#ERROR(TensorNetwork::appendTensorNetwork): Invalid request: " <<
   "Either primary or appended tensor network is not finalized!" << std::endl;
  return false;
 }
 //Check validity of leg pairing:
 auto * output0 = this->getTensorConn(0);
 assert(output0 != nullptr);
 auto output0_rank = output0->getNumLegs();
 auto * output1 = network.getTensorConn(0);
 assert(output1 != nullptr);
 auto output1_rank = output1->getNumLegs();
 if(output0_rank > 0 && output1_rank > 0){
  int ou0[output0_rank] = {0};
  int ou1[output1_rank] = {0};
  for(const auto & link: pairing){
   if(link.first >= output0_rank || link.second >= output1_rank){
    std::cout << "#ERROR(TensorNetwork::appendTensorNetwork): Invalid argument: Pairing: Out of bounds!" << std::endl;
    return false;
   }
   if(ou0[link.first]++ != 0){
    std::cout << "#ERROR(TensorNetwork::appendTensorNetwork): Invalid argument: Pairing: Repeated primary output leg!" << std::endl;
    return false;
   }
   if(ou1[link.second]++ != 0){
    std::cout << "#ERROR(TensorNetwork::appendTensorNetwork): Invalid argument: Pairing: Repeated secondary output leg!" << std::endl;
    return false;
   }
  }
 }else{
  if(pairing.size() > 0){
   std::cout << "#ERROR(TensorNetwork::appendTensorNetwork): Invalid argument: Pairing: Non-trivial pairing on scalar networks!" << std::endl;
   return false;
  }
 }
 //Pair output legs of the primary tensor network with output legs of the appended (secondary) tensor network:
 if(pairing.size() > 0){
  for(const auto & link: pairing){
   const auto & output0_leg_id = link.first;
   const auto & output1_leg_id = link.second;
   const auto & output0_leg = output0->getTensorLeg(output0_leg_id);
   const auto & output1_leg = output1->getTensorLeg(output1_leg_id);
   const auto input0_id = output0_leg.getTensorId();
   const auto input0_leg_id = output0_leg.getDimensionId();
   const auto input1_id = output1_leg.getTensorId();
   const auto input1_leg_id = output1_leg.getDimensionId();
   auto * input0 = this->getTensorConn(input0_id);
   assert(input0 != nullptr);
   auto * input1 = network.getTensorConn(input1_id);
   assert(input1 != nullptr);
   auto input0_leg = input0->getTensorLeg(input0_leg_id);
   input0_leg.resetTensorId(input1_id);
   input0_leg.resetDimensionId(input1_leg_id);
   input0->resetLeg(input0_leg_id,input0_leg);
   auto input1_leg = input1->getTensorLeg(input1_leg_id);
   input1_leg.resetTensorId(input0_id);
   input1_leg.resetDimensionId(input0_leg_id);
   input1->resetLeg(input1_leg_id,input1_leg);
  }
  //Delete matched legs from both output tensors:
  std::vector<unsigned int> matched_output_legs(pairing.size(),0);
  for(unsigned int i = 0; i < pairing.size(); ++i) matched_output_legs[i] = pairing[i].first;
  output0->deleteLegs(matched_output_legs);
  this->updateConnections(0);
  for(unsigned int i = 0; i < pairing.size(); ++i) matched_output_legs[i] = pairing[i].second;
  output1->deleteLegs(matched_output_legs);
  network.updateConnections(0);
 }
 //Append unmatched legs of the output tensor from the appended tensor network to the output tensor from the primary tensor network:
 output0_rank = output0->getNumLegs();
 output1_rank = output1->getNumLegs();
 for(unsigned int i = 0; i < output1_rank; ++i){
  output0->appendLeg(output1->getDimSpaceAttr(i),
                     output1->getDimExtent(i),
                     output1->getTensorLeg(i));
 }
 output0_rank = output0->getNumLegs();
 //Append all input tensors from the appended tensor network into the primary tensor network:
 auto tensors = network.getTensorConnAll();
 for(auto & tensor: tensors){
  unsigned int tensor_id = tensor->getTensorId();
  tensors_.emplace(std::make_pair(
                    tensor_id,*tensor
                   )
                  );
 }
 this->updateConnections(0); //update connections in just appended input tensors
 finalized_ = 1; //implicit leg pairing always keeps the primary tensor network in a finalized state
 return true;
}


bool TensorNetwork::reoderOutputModes(const std::vector<unsigned int> & order)
{
 if(finalized_ == 0){
  std::cout << "#ERROR(TensorNetwork::reorderOutputModes): Invalid request: " <<
   "Reodering modes in the output tensor of an unfinalized tensor network is forbidden!" << std::endl;
  return false;
 }
 auto * output_tensor = this->getTensorConn(0);
 assert(output_tensor != nullptr);
 auto output_tensor_rank = output_tensor->getNumLegs();
 if(order.size() != output_tensor_rank){
  std::cout << "#ERROR(TensorNetwork::reorderOutputModes): Invalid argument: Order: Wrong length: " <<
   order.size() << " versus " << output_tensor_rank << std::endl;
  return false;
 }
 if(output_tensor_rank > 0){
  auto legs = output_tensor->getTensorLegs();
  for(unsigned int i = 0; i < output_tensor_rank; ++i){
   const auto & old_leg_id = order[i];
   assert(old_leg_id < output_tensor_rank);
   output_tensor->resetLeg(i,legs[old_leg_id]);
  }
  updateConnections(0);
 }
 return true;
}


bool TensorNetwork::deleteTensor(unsigned int tensor_id)
{
 if(tensor_id == 0){
  std::cout << "#ERROR(TensorNetwork::deleteTensor): Invalid request: " <<
   "Deleting the output tensor of the tensor network is forbidden!" << std::endl;
  return false;
 }
 if(finalized_ == 0){
  std::cout << "#ERROR(TensorNetwork::deleteTensor): Invalid request: " <<
   "Deleting a tensor from an unfinalized tensor network is forbidden!" << std::endl;
  return false;
 }
 //Append the released legs from the deleted tensor to the output tensor:
 auto * tensor = this->getTensorConn(tensor_id);
 if(tensor == nullptr){
  std::cout << "#ERROR(TensorNetwork::deleteTensor): Invalid request: " <<
   "Tensor with id " << tensor_id << " is not found in the tensor network!" << std::endl;
  return false;
 }
 if(tensor->getNumLegs() > 0){
  auto * output_tensor = this->getTensorConn(0);
  assert(output_tensor != nullptr);
  auto output_tensor_rank = output_tensor->getNumLegs();
  const auto & legs = tensor->getTensorLegs();
  for(const auto & leg: legs){
   const auto other_tensor_id = leg.getTensorId();
   const auto other_tensor_leg_id = leg.getDimensionId();
   auto * other_tensor = this->getTensorConn(other_tensor_id);
   assert(other_tensor != nullptr);
   auto other_tensor_leg = other_tensor->getTensorLeg(other_tensor_leg_id);
   other_tensor_leg.resetTensorId(0);
   other_tensor_leg.resetDimensionId(output_tensor_rank);
   other_tensor->resetLeg(other_tensor_leg_id,other_tensor_leg);
   output_tensor->appendLeg(other_tensor->getDimSpaceAttr(other_tensor_leg_id),
                            other_tensor->getDimExtent(other_tensor_leg_id),
                            TensorLeg(
                             other_tensor_id,
                             other_tensor_leg_id,
                             reverseLegDirection(other_tensor_leg.getDirection())
                            )
                           );
   output_tensor_rank = output_tensor->getNumLegs();
  }
 }
 //Delete the tensor:
 assert(tensors_.erase(tensor_id) == 1);
 return true;
}


bool TensorNetwork::contractTensors(unsigned int left_id, unsigned int right_id, unsigned int result_id)
{
 if(left_id == right_id || left_id == result_id || right_id == result_id){
  std::cout << "#ERROR(TensorNetwork::contractTensors): Invalid arguments: Cannot be identical: " <<
   left_id << " " << right_id << " " << result_id << std::endl;
  return false;
 }
 if(left_id == 0 || right_id == 0 || result_id == 0){
  std::cout << "#ERROR(TensorNetwork::contractTensors): Invalid arguments: Output tensor #0 cannot participate: " <<
   left_id << " " << right_id << " " << result_id << std::endl;
  return false;
 }
 if(finalized_ == 0){
  std::cout << "#ERROR(TensorNetwork::contractTensors): Invalid request: " <<
   "Contracting tensors in an unfinalized tensor network is forbidden!" << std::endl;
  return false;
 }
 //`Finish
 return true;
}


bool TensorNetwork::buildFromTemplate(NetworkBuilder & builder)
{
 if(explicit_output_ == 0){
  std::cout << "#ERROR(TensorNetwork::buildFromTemplate): Invalid request: " <<
   "Trying to build a tensor network from a template while missing a full output tensor!" << std::endl;
  return false;
 }
 if(finalized_ != 0){
  std::cout << "#ERROR(TensorNetwork::buildFromTemplate): Invalid request: " <<
   "Trying to build a tensor network from a template while the tensor network is already finalized!" << std::endl;
  return false;
 }
 builder.build(*this);
 return true;
}

} //namespace numerics

} //namespace exatn
