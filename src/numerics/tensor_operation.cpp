/** ExaTN::Numerics: Tensor operation
REVISION: 2020/04/22

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_operation.hpp"

#include <iostream>

namespace exatn{

namespace numerics{

TensorOperation::TensorOperation(TensorOpCode opcode,
                                 unsigned int num_operands,
                                 unsigned int num_scalars,
                                 std::size_t mutability):
 num_operands_(num_operands), num_scalars_(num_scalars),
 mutation_(mutability), opcode_(opcode), id_(0),
 scalars_(num_scalars,std::complex<double>{0.0,0.0})
{
 operands_.reserve(num_operands);
}

void TensorOperation::printIt() const
{
 std::cout << "TensorOperation(" << static_cast<int>(opcode_) << "){" << std::endl;
 if(pattern_.length() > 0) std::cout << " " << pattern_ << std::endl;
 for(const auto & operand: operands_){
  const auto & tensor = std::get<0>(operand);
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
  const auto & tensor = std::get<0>(operand);
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

unsigned int TensorOperation::getNumOperandsOut() const
{
 std::size_t n = 0, mut = mutation_, bit0 = 1;
 for(unsigned int i = 0; i < num_operands_; ++i){
  n += (mut & bit0);
  mut >>= 1;
 }
 return static_cast<unsigned int>(n);
}

unsigned int TensorOperation::getNumOperandsSet() const
{
 return static_cast<unsigned int>(operands_.size());
}

TensorHashType TensorOperation::getTensorOperandHash(unsigned int op_num) const
{
 return this->getTensorOperand(op_num)->getTensorHash();
}

bool TensorOperation::operandIsConjugated(unsigned int op_num) const
{
 assert(op_num < operands_.size());
 return std::get<1>(operands_[op_num]);
}

bool TensorOperation::operandIsMutable(unsigned int op_num) const
{
 assert(op_num < operands_.size());
 return std::get<2>(operands_[op_num]);
}

std::shared_ptr<Tensor> TensorOperation::getTensorOperand(unsigned int op_num,
                                                          bool * conjugated,
                                                          bool * mutated) const
{
 if(op_num < operands_.size()){
  if(conjugated != nullptr) *conjugated = std::get<1>(operands_[op_num]);
  if(mutated != nullptr) *mutated = std::get<2>(operands_[op_num]);
  return std::get<0>(operands_[op_num]);
 }
 return std::shared_ptr<Tensor>(nullptr);
}

void TensorOperation::setTensorOperand(std::shared_ptr<Tensor> tensor,
                                       bool conjugated,
                                       bool mutated)
{
 assert(tensor);
 assert(operands_.size() < num_operands_);
 operands_.emplace_back(std::make_tuple(tensor,conjugated,mutated));
 return;
}

void TensorOperation::setTensorOperand(std::shared_ptr<Tensor> tensor,
                                       bool conjugated)
{
 return this->setTensorOperand(tensor,conjugated,(mutation_>>operands_.size())&(0x1U));
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
 if(operands_.size() == num_operands_ && scalars_.size() == num_scalars_){
  pattern_ = pattern;
 }else{
  std::cout << "#ERROR(exatn::numerics::TensorOperation::setIndexPattern): "
            << "Index pattern cannot be set until all operands and scalars have been set!\n";
  assert(false);
 }
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

TensorHashType TensorOperation::getTensorOpHash() const
{
 return reinterpret_cast<TensorHashType>((void*)this);
}

} //namespace numerics

} //namespace exatn
