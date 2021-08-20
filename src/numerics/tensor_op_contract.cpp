/** ExaTN::Numerics: Tensor operation: Contracts two tensors and accumulates the result into another tensor
REVISION: 2021/08/20

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_contract.hpp"

#include "tensor_node_executor.hpp"

#include <cmath>

namespace exatn{

namespace numerics{

TensorOpContract::TensorOpContract():
 TensorOperation(TensorOpCode::CONTRACT,3,2,1+0*2+0*4,{0,1,2}),
 accumulative_(true)
{
 this->setScalar(0,std::complex<double>{1.0,0.0}); //default alpha prefactor
 this->setScalar(1,std::complex<double>{1.0,0.0}); //default beta prefactor (accumulative)
}

bool TensorOpContract::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands() && this->getIndexPattern().length() > 0);
}

int TensorOpContract::accept(runtime::TensorNodeExecutor & node_executor,
                             runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

double TensorOpContract::getFlopEstimate() const
{
 if(this->isSet()){
  double vol0 = static_cast<double>(this->getTensorOperand(0)->getVolume());
  double vol1 = static_cast<double>(this->getTensorOperand(1)->getVolume());
  double vol2 = static_cast<double>(this->getTensorOperand(2)->getVolume());
  return std::sqrt(vol0*vol1*vol2); //FMA flops (without FMA factor)
 }
 return 0.0;
}

std::unique_ptr<TensorOperation> TensorOpContract::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpContract());
}

void TensorOpContract::resetAccumulative(bool accum)
{
 accumulative_ = accum;
 return;
}

DimExtent TensorOpContract::getCombinedDimExtent(IndexKind index_kind) const
{
 assert(index_info_);
 DimExtent dim_ext = 1;
 switch(index_kind){
  case IndexKind::LEFT:
   for(const auto & ind: index_info_->left_indices_) dim_ext *= getTensorOperand(0)->getDimExtent(ind.arg_pos[0]);
   break;
  case IndexKind::RIGHT:
   for(const auto & ind: index_info_->right_indices_) dim_ext *= getTensorOperand(0)->getDimExtent(ind.arg_pos[0]);
   break;
  case IndexKind::CONTR:
   for(const auto & ind: index_info_->contr_indices_) dim_ext *= getTensorOperand(1)->getDimExtent(ind.arg_pos[1]);
   break;
  case IndexKind::HYPER:
   for(const auto & ind: index_info_->hyper_indices_) dim_ext *= getTensorOperand(0)->getDimExtent(ind.arg_pos[0]);
   break;
  default:
   assert(false);
 }
 return dim_ext;
}

void TensorOpContract::determineNumBisections(unsigned int num_processes,
                                              std::size_t mem_per_process,
                                              unsigned int * bisect_left,
                                              unsigned int * bisect_right,
                                              unsigned int * bisect_contr,
                                              unsigned int * bisect_hyper) const
{
 *bisect_left = 0;
 *bisect_right = 0;
 *bisect_contr = 0;
 *bisect_hyper = 0;

 const auto tens_elem_size = TensorElementTypeSize(getTensorOperand(2)->getElementType());
 auto dim_left = getCombinedDimExtent(IndexKind::LEFT);
 auto dim_right = getCombinedDimExtent(IndexKind::RIGHT);
 auto dim_contr = getCombinedDimExtent(IndexKind::CONTR);
 auto dim_hyper = getCombinedDimExtent(IndexKind::HYPER);
 std::size_t proc_count = num_processes;
 std::size_t mem_count = mem_per_process * num_processes;

 auto dfs = [](std::size_t mem_lim, std::size_t tens_size){
  return (tens_size > static_cast<std::size_t>(static_cast<double>(mem_lim)*replication_threshold));
 };

 while(proc_count > 1){
  bool serial = false;
  std::size_t tensor_size = 0;
  if(dim_hyper > 1){
   serial = false;
   dim_hyper /= 2;
   (*bisect_hyper)++;
  }else{
   if(dim_right >= dim_left && dim_right >= dim_contr){
    tensor_size = dim_left * dim_contr * tens_elem_size;
    serial = dfs(mem_count,tensor_size);
    dim_right /= 2;
    (*bisect_right)++;
   }else if(dim_left >= dim_contr && dim_left >= dim_right){
    tensor_size = dim_contr * dim_right * tens_elem_size;
    serial = dfs(mem_count,tensor_size);
    dim_left /= 2;
    (*bisect_left)++;
   }else if(dim_contr >= dim_right && dim_contr >= dim_left){
    tensor_size = dim_right * dim_left * tens_elem_size;
    serial = dfs(mem_count,tensor_size);
    dim_contr /= 2;
    (*bisect_contr)++;
   }
  }
  if(!serial){
   mem_count -= tensor_size;
   mem_count /= 2;
   proc_count /= 2;
  }
 }
 return;
}

void TensorOpContract::introduceOptTemporaries(unsigned int num_processes,
                                               std::size_t mem_per_process,
                                               const std::vector<PosIndexLabel> & left_indices,
                                               const std::vector<PosIndexLabel> & right_indices,
                                               const std::vector<PosIndexLabel> & contr_indices,
                                               const std::vector<PosIndexLabel> & hyper_indices)
{
 //std::cout << "#DEBUG(TensorOpContract::introduceOptTemporaries): Parallel parameters: "
 // << num_processes << " " << mem_per_process << std::endl; //debug
 assert(num_processes > 0);
 assert((num_processes & (num_processes - 1)) == 0);
 assert(mem_per_process > 0);
 bool success = true;
 index_info_ = std::make_shared<IndexInfo>(left_indices,right_indices,contr_indices,hyper_indices);
 unsigned int bisect_left = 0, bisect_right = 0, bisect_contr = 0, bisect_hyper = 0;
 determineNumBisections(num_processes,mem_per_process,&bisect_left,&bisect_right,&bisect_contr,&bisect_hyper);
 //std::cout << "#DEBUG(TensorOpContract::introduceOptTemporaries): Bisections: "
 // << bisect_left << " " << bisect_right << " " << bisect_contr << " " << bisect_hyper << std::endl; //debug
 //Set the splitting depth for the left indices:
 unsigned int split_left = 0;
 if(bisect_left > 0){
  auto & indices = index_info_->left_indices_;
  int i = indices.size();
  auto n = bisect_left;
  while(n-- > 0){
   if(--i < 0) i = (indices.size() - 1);
   const unsigned int new_depth = indices[i].depth + 1;
   if(getTensorOperand(0)->getDimExtent(indices[i].arg_pos[0]) >= (1U << new_depth)){
    if((indices[i].depth)++ == 0) ++split_left;
   }
  }
 }
 //Set the splitting depth for the right indices:
 unsigned int split_right = 0;
 if(bisect_right > 0){
  auto & indices = index_info_->right_indices_;
  int i = indices.size();
  auto n = bisect_right;
  while(n-- > 0){
   if(--i < 0) i = (indices.size() - 1);
   const unsigned int new_depth = indices[i].depth + 1;
   if(getTensorOperand(0)->getDimExtent(indices[i].arg_pos[0]) >= (1U << new_depth)){
    if((indices[i].depth)++ == 0) ++split_right;
   }
  }
 }
 //Set the splitting depth for the contracted indices:
 unsigned int split_contr = 0;
 if(bisect_contr > 0){
  auto & indices = index_info_->contr_indices_;
  int i = indices.size();
  auto n = bisect_contr;
  while(n-- > 0){
   if(--i < 0) i = (indices.size() - 1);
   const unsigned int new_depth = indices[i].depth + 1;
   if(getTensorOperand(1)->getDimExtent(indices[i].arg_pos[1]) >= (1U << new_depth)){
    if((indices[i].depth)++ == 0) ++split_contr;
   }
  }
 }
 //Set the splitting depth for the hyper indices:
 unsigned int split_hyper = 0;
 if(bisect_hyper > 0){
  auto & indices = index_info_->hyper_indices_;
  int i = indices.size();
  auto n = bisect_hyper;
  while(n-- > 0){
   if(--i < 0) i = (indices.size() - 1);
   const unsigned int new_depth = indices[i].depth + 1;
   if(getTensorOperand(0)->getDimExtent(indices[i].arg_pos[0]) >= (1U << new_depth)){
    if((indices[i].depth)++ == 0) ++split_hyper;
   }
  }
 }
 //index_info_->printIt(); //debug
 //Replace original tensor operands with new ones with proper decomposition:
 //Destination tensor operand:
 if(split_left > 0 || split_right > 0 || split_hyper > 0){
  std::vector<std::pair<unsigned int, unsigned int>> split_dims(split_left+split_right+split_hyper);
  unsigned int i = 0;
  for(const auto & ind: index_info_->left_indices_){
   if(ind.depth > 0) split_dims[i++] = std::pair<unsigned int, unsigned int>(ind.arg_pos[0],ind.depth);
  }
  for(const auto & ind: index_info_->right_indices_){
   if(ind.depth > 0) split_dims[i++] = std::pair<unsigned int, unsigned int>(ind.arg_pos[0],ind.depth);
  }
  for(const auto & ind: index_info_->hyper_indices_){
   if(ind.depth > 0) split_dims[i++] = std::pair<unsigned int, unsigned int>(ind.arg_pos[0],ind.depth);
  }
  auto original_tensor = getTensorOperand(0); assert(original_tensor);
  if(original_tensor->isComposite()){
   auto success = resetTensorOperand(0,makeSharedTensorComposite(split_dims,*original_tensor)); assert(success);
  }
 }
 //Left tensor operand:
 if(split_contr > 0 || split_left > 0 || split_hyper > 0){
  std::vector<std::pair<unsigned int, unsigned int>> split_dims(split_contr+split_left+split_hyper);
  unsigned int i = 0;
  for(const auto & ind: index_info_->contr_indices_){
   if(ind.depth > 0) split_dims[i++] = std::pair<unsigned int, unsigned int>(ind.arg_pos[1],ind.depth);
  }
  for(const auto & ind: index_info_->left_indices_){
   if(ind.depth > 0) split_dims[i++] = std::pair<unsigned int, unsigned int>(ind.arg_pos[1],ind.depth);
  }
  for(const auto & ind: index_info_->hyper_indices_){
   if(ind.depth > 0) split_dims[i++] = std::pair<unsigned int, unsigned int>(ind.arg_pos[1],ind.depth);
  }
  auto original_tensor = getTensorOperand(1); assert(original_tensor);
  if(original_tensor->isComposite()){
   auto success = resetTensorOperand(1,makeSharedTensorComposite(split_dims,*original_tensor)); assert(success);
  }
 }
 //Right tensor operand:
 if(split_contr > 0 || split_right > 0 || split_hyper > 0){
  std::vector<std::pair<unsigned int, unsigned int>> split_dims(split_contr+split_right+split_hyper);
  unsigned int i = 0;
  for(const auto & ind: index_info_->contr_indices_){
   if(ind.depth > 0) split_dims[i++] = std::pair<unsigned int, unsigned int>(ind.arg_pos[2],ind.depth);
  }
  for(const auto & ind: index_info_->right_indices_){
   if(ind.depth > 0) split_dims[i++] = std::pair<unsigned int, unsigned int>(ind.arg_pos[2],ind.depth);
  }
  for(const auto & ind: index_info_->hyper_indices_){
   if(ind.depth > 0) split_dims[i++] = std::pair<unsigned int, unsigned int>(ind.arg_pos[2],ind.depth);
  }
  auto original_tensor = getTensorOperand(2); assert(original_tensor);
  if(original_tensor->isComposite()){
   auto success = resetTensorOperand(2,makeSharedTensorComposite(split_dims,*original_tensor)); assert(success);
  }
 }
 return;
}

void TensorOpContract::introduceOptTemporaries(unsigned int num_processes, std::size_t mem_per_process)
{
 std::vector<std::string> tensors;
 std::vector<PosIndexLabel> left_inds, right_inds, contr_inds, hyper_inds;
 auto parsed = parse_tensor_contraction(getIndexPattern(),tensors,left_inds,right_inds,contr_inds,hyper_inds);
 if(!parsed){
  std::cout << "#ERROR(TensorOpContract:introduceOptTemporaries): Invalid tensor contraction specification: "
            << getIndexPattern() << std::endl << std::flush;
  assert(false);
 }
 return introduceOptTemporaries(num_processes,mem_per_process,left_inds,right_inds,contr_inds,hyper_inds);
}

std::size_t TensorOpContract::decompose(const TensorMapper & tensor_mapper)
{
 if(this->isComposite()){
  if(simple_operations_.empty()){
   //Identify parallel configuration:
   const auto num_procs = tensor_mapper.getNumProcesses();
   const auto proc_rank = tensor_mapper.getProcessRank();
   const auto & intra_comm = tensor_mapper.getMPICommProxy();
   //Proceed with decomposition:
   assert(index_info_);
   
   std::abort(); //`debug
  }
 }
 return simple_operations_.size();
}

} //namespace numerics

} //namespace exatn
