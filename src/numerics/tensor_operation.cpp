/** ExaTN::Numerics: Tensor operation
REVISION: 2019/10/13

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_operation.hpp"

#include <iostream>

namespace exatn{

namespace numerics{

TensorOperation::TensorOperation(TensorOpCode opcode, unsigned int num_operands, unsigned int num_scalars):
 num_operands_(num_operands), num_scalars_(num_scalars), opcode_(opcode), id_(0),
 scalars_(num_scalars,std::complex<double>{0.0,0.0})
{
 operands_.reserve(num_operands);
}

void TensorOperation::printIt() const
{
 std::cout << "TensorOperation(" << static_cast<int>(opcode_) << "){" << std::endl;
 if(pattern_.length() > 0) std::cout << " " << pattern_ << std::endl;
 for(const auto & operand: operands_){
  const auto & tensor = operand.first;
  std::cout << " ";
  tensor->printIt();
  std::cout << std::endl;
 }
 for(const auto & scalar: scalars_){
  std::cout << " " << scalar;
 }
 if(scalars_.size() > 0) std::cout << std::endl;
 std::cout << "}" << std::endl;
 return;
}

void TensorOperation::printItFile(std::ofstream & output_file) const
{
 output_file << "TensorOperation(" << static_cast<int>(opcode_) << "){" << std::endl;
 if(pattern_.length() > 0) output_file << " " << pattern_ << std::endl;
 for(const auto & operand: operands_){
  const auto & tensor = operand.first;
  output_file << " ";
  tensor->printItFile(output_file);
  output_file << std::endl;
 }
 for(const auto & scalar: scalars_){
  output_file << " " << scalar;
 }
 if(scalars_.size() > 0) output_file << std::endl;
 output_file << "}" << std::endl;
 output_file.flush();
 return;
}

TensorOpCode TensorOperation::getOpcode() const
{
 return opcode_;
}

unsigned int TensorOperation::getNumOperands() const
{
 return num_operands_;
}

unsigned int TensorOperation::getNumOperandsSet() const
{
 return static_cast<unsigned int>(operands_.size());
}

TensorHashType TensorOperation::getTensorOperandHash(unsigned int op_num) const
{
 return this->getTensorOperand(op_num)->getTensorHash();
}

std::shared_ptr<Tensor> TensorOperation::getTensorOperand(unsigned int op_num, bool * conjugated) const
{
 if(op_num < operands_.size()){
  if(conjugated != nullptr) *conjugated = operands_[op_num].second;
  return operands_[op_num].first;
 }
 return std::shared_ptr<Tensor>(nullptr);
}

void TensorOperation::setTensorOperand(std::shared_ptr<Tensor> tensor, bool conjugated)
{
 assert(tensor);
 assert(operands_.size() < num_operands_);
 operands_.emplace_back(std::make_pair(tensor,conjugated));
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

void TensorOperation::setId(std::size_t id)
{
 id_ = id;
 return;
}

std::size_t TensorOperation::getId() const
{
 return id_;
}

} //namespace numerics

} //namespace exatn
