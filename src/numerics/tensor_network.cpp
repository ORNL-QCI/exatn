/** ExaTN::Numerics: Tensor network
REVISION: 2019/07/08

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_network.hpp"

#include <iostream>
#include <assert.h>

namespace exatn{

namespace numerics{

TensorNetwork::TensorNetwork(const std::string & name):
name_(name)
{
 tensors_.emplace( //output tensor (id = 0)
           std::make_pair(
            0U,
            TensorConn(std::make_shared<Tensor>(name),0U,std::vector<TensorLeg>())
           )
          );
}

TensorNetwork::TensorNetwork()
{
 tensors_.emplace( //output tensor (id = 0)
           std::make_pair(
            0U,
            TensorConn(std::make_shared<Tensor>("_SMOKY_TENSOR_"),0U,std::vector<TensorLeg>())
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
 return (tensors_.size() == 1); //only output tensor exists => empty
}

unsigned int TensorNetwork::getNumTensors() const
{
 return static_cast<unsigned int>(tensors_.size() - 1); //output tensor is not counted
}

const std::string & TensorNetwork::getName() const
{
 return name_;
}

const TensorConn * TensorNetwork::getTensorConn(unsigned int tensor_id) const
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

bool TensorNetwork::appendTensor(unsigned int tensor_id,                     //in: tensor id (unique within the tensor network)
                                 std::shared_ptr<Tensor> tensor,             //in: appended tensor
                                 const std::vector<TensorLeg> & connections) //in: tensor connections
{
 //`Finish
 return true;
}

bool TensorNetwork::appendTensor(unsigned int tensor_id,                                             //in: tensor id (unique within the tensor network)
                                 std::shared_ptr<Tensor> tensor,                                     //in: appended tensor
                                 const std::vector<std::pair<unsigned int, unsigned int>> & pairing) //in: leg pairing: output tensor mode -> appended tensor mode
{
 //`Finish
 return true;
}

bool TensorNetwork::appendTensorNetwork(TensorNetwork && network,                                           //in: appended tensor network
                                        const std::vector<std::pair<unsigned int, unsigned int>> & pairing) //in: leg pairing: output tensor mode (primary) -> output tensor mode (appended)
{
 //`Finish
 return true;
}

void TensorNetwork::reoderOutputModes(const std::vector<unsigned int> & order)
{
 //`Finish
 return;
}

bool TensorNetwork::deleteTensor(unsigned int tensor_id)
{
 assert(tensor_id != 0); //output tensor cannot be deleted
 //`Finish
 return true;
}

bool TensorNetwork::contractTensors(unsigned int left_id, unsigned int right_id, unsigned int result_id)
{
 assert(left_id != right_id && left_id != result_id && right_id != result_id);
 //`Finish
 return true;
}

} //namespace numerics

} //namespace exatn
