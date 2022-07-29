/** ExaTN::Numerics: Tensor operation
REVISION: 2022/07/29

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_operation.hpp"
#include "tensor_symbol.hpp"

#include <iostream>
#include <ios>

namespace exatn{

namespace numerics{

TensorOperation::TensorOperation(TensorOpCode opcode,
                                 unsigned int num_operands,
                                 unsigned int num_scalars,
                                 std::size_t mutability,
                                 std::initializer_list<int> symbolic_positions):
 symb_pos_(symbolic_positions), num_operands_(num_operands), num_scalars_(num_scalars),
 mutation_(mutability), opcode_(opcode), id_(0), repeatable_(true),
 scalars_(num_scalars,std::complex<double>{0.0,0.0})
{
 operands_.reserve(num_operands);
}

bool TensorOperation::isComposite() const
{
 bool is_composite = this->isSet();
 assert(is_composite);
 is_composite = false;
 for(const auto & operand: operands_) is_composite = (is_composite || std::get<0>(operand)->isComposite());
 return is_composite;
}

std::shared_ptr<TensorOperation> TensorOperation::operator[](std::size_t operation_id)
{
 assert(operation_id < simple_operations_.size());
 return simple_operations_[operation_id];
}

double TensorOperation::getFlopEstimate() const
{
 return 0.0; //flop estimate is not available by default
}

double TensorOperation::getWordEstimate() const
{
 double total_volume = 0.0;
 if(this->isSet()){
  for(unsigned int i = 0; i < this->getNumOperands(); ++i){
   total_volume += static_cast<double>(this->getTensorOperand(i)->getVolume());
  }
 }
 return total_volume;
}

void TensorOperation::printIt() const
{
 std::cout << "TensorOperation(opcode=" << static_cast<int>(opcode_) << ")[id=" << id_ << "]{" << std::endl;
 if(pattern_.length() > 0) std::cout << " " << pattern_ << std::endl;
 for(const auto & operand: operands_){
  const auto & tensor = std::get<0>(operand);
  if(tensor != nullptr){
   std::cout << " ";
   tensor->printIt();
   std::cout << std::endl;
  }else{
   std::cout << "#ERROR(exatn::TensorOperation::printIt): Tensor operand is NULL!" << std::endl << std::flush;
   assert(false);
  }
 }
 for(const auto & scalar: scalars_){
  std::cout << " " << scalar;
 }
 if(scalars_.size() > 0) std::cout << std::endl;
 std::cout << " GFlop estimate = " << std::scientific << this->getFlopEstimate()/1e9 << std::endl;
 std::cout << " GWord estimate = " << std::scientific << this->getWordEstimate()/1e9 << std::endl;
 std::cout << "}" << std::endl << std::flush;
 return;
}

void TensorOperation::printItFile(std::ofstream & output_file) const
{
 output_file << "TensorOperation(opcode=" << static_cast<int>(opcode_) << ")[id=" << id_ << "]{" << std::endl;
 if(pattern_.length() > 0) output_file << " " << pattern_ << std::endl;
 for(const auto & operand: operands_){
  const auto & tensor = std::get<0>(operand);
  if(tensor != nullptr){
   output_file << " ";
   tensor->printItFile(output_file);
   output_file << std::endl;
  }else{
   std::cout << "#ERROR(exatn::TensorOperation::printItFile): Tensor operand is NULL!" << std::endl << std::flush;
   assert(false);
  }
 }
 for(const auto & scalar: scalars_){
  output_file << " " << scalar;
 }
 if(scalars_.size() > 0) output_file << std::endl;
 output_file << " GFlop estimate = " << std::scientific << this->getFlopEstimate()/1e9 << std::endl;
 output_file << " GWord estimate = " << std::scientific << this->getWordEstimate()/1e9 << std::endl;
 output_file << "}" << std::endl;
 //output_file.flush();
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

unsigned int TensorOperation::getTensorOperandId(unsigned int op_num) const
{
 assert(op_num < operand_tensor_ids_.size());
 return operand_tensor_ids_[op_num];
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

void TensorOperation::appendTensorOperand(std::shared_ptr<Tensor> tensor,
                                          bool conjugated,
                                          bool mutated)
{
 assert(tensor);
 assert(operands_.size() < num_operands_);
 operands_.emplace_back(std::make_tuple(tensor,conjugated,mutated));
 return;
}

void TensorOperation::setTensorOperand(std::shared_ptr<Tensor> tensor,
                                       bool conjugated,
                                       unsigned int tensor_id)
{
 operand_tensor_ids_.emplace_back(tensor_id);
 return this->appendTensorOperand(tensor,conjugated,(mutation_>>operands_.size())&(0x1U));
}

bool TensorOperation::resetTensorOperand(unsigned int op_num,
                                         std::shared_ptr<Tensor> tensor)
{
 assert(tensor);
 if(op_num >= this->getNumOperandsSet()) return false;
 std::get<0>(operands_[op_num]) = tensor;
 return true;
}

void TensorOperation::dissociateTensorOperands()
{
 if(!repeatable_){
  for(auto & oprnd: operands_){
   //std::cout << "#DEBUG: Dissociating " << std::get<0>(oprnd)->getName()
   //          << " with use_count " << std::get<0>(oprnd).use_count() << std::endl; //debug
   std::get<0>(oprnd).reset();
  }
 }
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

std::string TensorOperation::getIndexPatternReduced() const
{
 std::string reduced;
 if(pattern_.length() > 0){
  const auto num_operands = this->getNumOperands();
  std::vector<std::string> tensors;
  auto parsed = parse_tensor_network(pattern_,tensors);
  if(parsed){
   const auto num_tensors = tensors.size();
   assert(num_tensors == num_operands);
   for(unsigned int oprnd = 0; oprnd < num_operands; ++oprnd){
    const auto & tensor = *(this->getTensorOperand(oprnd));
    if(symb_pos_[oprnd] >= 0){ //tensor operand is present in the symbolic index pattern
     std::string tensor_name;
     std::vector<IndexLabel> indices;
     bool conj;
     parsed = parse_tensor(tensors[symb_pos_[oprnd]],tensor_name,indices,conj);
     if(parsed){
      unsigned int i = 0;
      auto iter = indices.begin();
      while(iter != indices.end()){
       if(tensor.getDimExtent(i++) > 1){
        ++iter;
       }else{
        iter = indices.erase(iter); //remove indices associated with extent-1 dimensions
       }
      }
      tensors[symb_pos_[oprnd]] = assemble_symbolic_tensor(tensor_name,indices,conj);
     }else{
      std::cout << "#ERROR(exatn::numerics::TensorOperation::getIndexPatternReduced): "
                << "Unable to parse tensor operand " << symb_pos_[oprnd]
                << " in symbolic tensor operation specification: " << pattern_ << std::endl;
      assert(false);
     }
    }
   }
   if(opcode_ == TensorOpCode::DECOMPOSE_SVD3){ //delete the middle tensor from the symbolic specification
    tensors.erase(tensors.begin()+2);
   }
   reduced = assemble_symbolic_tensor_network(tensors);
  }else{
   std::cout << "#ERROR(exatn::numerics::TensorOperation::getIndexPatternReduced): "
             << "Unable to parse the symbolic tensor operation specification: "
             << pattern_ << std::endl;
   assert(false);
  }
 }
 return std::move(reduced);
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
