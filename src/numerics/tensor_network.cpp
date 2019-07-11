/** ExaTN::Numerics: Tensor network
REVISION: 2019/07/10

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
 return (tensors_.size() <= 1); //only output tensor exists => empty
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


std::shared_ptr<Tensor> TensorNetwork::getTensor(unsigned int tensor_id)
{
 auto it = tensors_.find(tensor_id);
 if(it == tensors_.end()) return std::shared_ptr<Tensor>(nullptr);
 return (it->second).getTensor();
}


bool TensorNetwork::finalize()
{
 if(this->isEmpty()){ //empty networks cannot be finalized
  std::cout << "#ERROR(TensorNetwork::finalize): Empty tensor network cannot be finalized!" << std::endl;
  return false;
 }
 finalized_ = 1;
 return true;
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
  if(tensconn == nullptr){
   std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid argument: Connections are invalid!" << std::endl;
   return false;
  }
  const auto & tens_legs = tensconn->getTensorLegs();
  const auto & tens_leg = tens_legs[leg.getDimensionId()];
  if(tens_leg.getTensorId() != tensor_id || tens_leg.getDimensionId() != mode){
   std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid argument: Connections are invalid!" << std::endl;
   return false;
  }
  ++mode;
 }
 //Append the tensor to the tensor network:
 auto new_pos = tensors_.emplace(
                                 std::pair<unsigned int, TensorConn>(
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
 //Check leg pairing:
 bool dir_present = (leg_dir.size() > 0);
 auto * output = this->getTensorConn(0); assert(output != nullptr);
 auto output_rank = output->getNumLegs();
 auto tensor_rank = tensor->getRank();
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
 }
 //`Finish
 finalized_ = 1;
 return true;
}


bool TensorNetwork::appendTensorNetwork(TensorNetwork && network,                                           //in: appended tensor network
                                        const std::vector<std::pair<unsigned int, unsigned int>> & pairing) //in: leg pairing: output tensor mode (primary) -> output tensor mode (appended)
{
 //`Finish
 finalized_ = 1;
 return true;
}


bool TensorNetwork::reoderOutputModes(const std::vector<unsigned int> & order)
{
 //`Finish
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
 //`Finish
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

} //namespace numerics

} //namespace exatn
