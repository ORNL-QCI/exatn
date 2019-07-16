/** ExaTN::Numerics: Tensor connected to other tensors inside a tensor network
REVISION: 2019/07/15

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_connected.hpp"

#include <iostream>
#include <assert.h>

namespace exatn{

namespace numerics{

TensorConn::TensorConn(std::shared_ptr<Tensor> tensor, unsigned int id, const std::vector<TensorLeg> & legs):
 tensor_(tensor), id_(id), legs_(legs)
{
}

void TensorConn::printIt() const
{
 std::cout << id_ << ": ";
 tensor_->printIt();
 std::cout << ": { ";
 for(const auto & leg: legs_) leg.printIt();
 std::cout << " }" << std::endl;
 return;
}

unsigned int TensorConn::getNumLegs() const
{
 return tensor_->getRank();
}

std::shared_ptr<Tensor> TensorConn::getTensor()
{
 return tensor_;
}

unsigned int TensorConn::getTensorId() const
{
 return id_;
}

const TensorLeg & TensorConn::getTensorLeg(unsigned int leg_id) const
{
 assert(leg_id < legs_.size());
 return legs_[leg_id];
}

const std::vector<TensorLeg> & TensorConn::getTensorLegs() const
{
 return legs_;
}

DimExtent TensorConn::getDimExtent(unsigned int dim_id) const
{
 return tensor_->getDimExtent(dim_id);
}

std::pair<SpaceId,SubspaceId> TensorConn::getDimSpaceAttr(unsigned int dim_id) const
{
 return tensor_->getDimSpaceAttr(dim_id);
}

void TensorConn::resetLeg(unsigned int leg_id, TensorLeg tensor_leg)
{
 assert(leg_id < legs_.size());
 legs_[leg_id].resetConnection(tensor_leg.getTensorId(),
                               tensor_leg.getDimensionId(),
                               tensor_leg.getDirection());
 return;
}

void TensorConn::deleteLeg(unsigned int leg_id)
{
 assert(leg_id < legs_.size());
 legs_.erase(legs_.cbegin()+leg_id);
 tensor_->deleteDimension(leg_id);
 return;
}

void TensorConn::appendLeg(std::pair<SpaceId,SubspaceId> subspace, DimExtent dim_extent, TensorLeg tensor_leg)
{
 tensor_->appendDimension(subspace,dim_extent);
 legs_.emplace_back(tensor_leg);
 return;
}

void TensorConn::appendLeg(DimExtent dim_extent, TensorLeg tensor_leg)
{
 this->appendLeg(std::pair<SpaceId,SubspaceId>{SOME_SPACE,0},dim_extent,tensor_leg);
 return;
}

} //namespace numerics

} //namespace exatn
