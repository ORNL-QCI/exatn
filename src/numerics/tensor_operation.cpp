/** ExaTN::Numerics: Tensor operation
REVISION: 2019/05/30

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_operation.hpp"

namespace exatn{

namespace numerics{

TensorOperation::TensorOperation(unsigned int num_operands, unsigned int num_scalars):
 num_operands_(num_operands), num_scalars_(num_scalars), scalars_(num_scalars,std::complex<double>{0.0,0.0})
{
 operands_.reserve(num_operands);
}

unsigned int TensorOperation::getNumOperands() const
{
 return num_operands_;
}

unsigned int TensorOperation::getNumOperandsSet() const
{
 return static_cast<unsigned int>(operands_.size());
}

const Tensor * TensorOperation::getTensorOperand(unsigned int op_num) const
{
 if(op_num < operands_.size()) return operands_[op_num];
 return nullptr;
}

void TensorOperation::setTensorOperand(const Tensor * tensor)
{
 assert(tensor != nullptr);
 assert(operands_.size() < num_operands_);
 operands_.push_back(tensor);
 return;
}

unsigned int TensorOperation::getNumScalars() const
{
 return num_scalars_;
}

unsigned int TensorOperation::getNumScalarsSet() const
{
 return static_cast<unsigned int>(scalars_.size());
}

std::complex<double> TensorOperation::getScalar(unsigned int scalar_num) const
{
 assert(scalar_num < scalars_.size());
 return scalars_[scalar_num];
}

void TensorOperation::setScalar(unsigned int scalar_num, const std::complex<double> scalar)
{
 assert(scalar_num < scalars_.size());
 scalars_[scalar_num] = scalar;
 return;
}

const std::string & TensorOperation::getIndexPattern() const
{
 return pattern_;
}

void TensorOperation::setIndexPattern(const std::string & pattern)
{
 assert(operands_.size() == num_operands_ && scalars_.size() == num_scalars_);
 pattern_ = pattern;
 return;
}

} //namespace numerics

} //namespace exatn
