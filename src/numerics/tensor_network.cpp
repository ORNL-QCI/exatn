/** ExaTN::Numerics: Tensor network
REVISION: 2019/07/02

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
 tensors_.emplace(
           std::make_pair(
            0U,
            TensorConn(std::make_shared<Tensor>(name),0U,std::vector<TensorLeg>())
           )
          );
}

TensorNetwork::TensorNetwork()
{
 tensors_.emplace(
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

unsigned int TensorNetwork::getNumTensors() const
{
 return static_cast<unsigned int>(tensors_.size() - 1); //output tensor is not counted
}

} //namespace numerics

} //namespace exatn
