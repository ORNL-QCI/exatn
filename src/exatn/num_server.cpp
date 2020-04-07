/** ExaTN::Numerics: Numerical server
REVISION: 2020/04/07

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "num_server.hpp"

#include <vector>
#include <list>
#include <map>
#include <future>

#ifdef MPI_ENABLED
#include "mpi.h"
#endif

#include <cassert>

namespace exatn{

/** Numerical server (singleton) **/
std::shared_ptr<NumServer> numericalServer {nullptr}; //initialized by exatn::initialize()


#ifdef MPI_ENABLED
NumServer::NumServer(const MPICommProxy & communicator,
                     const std::string & graph_executor_name,
                     const std::string & node_executor_name):
 contr_seq_optimizer_("dummy"),
 tensor_rt_(std::make_shared<runtime::TensorRuntime>(communicator,graph_executor_name,node_executor_name))
{
 int mpi_error = MPI_Comm_size(*(communicator.get<MPI_Comm>()),&num_processes_); assert(mpi_error == MPI_SUCCESS);
 mpi_error = MPI_Comm_rank(*(communicator.get<MPI_Comm>()),&process_rank_); assert(mpi_error == MPI_SUCCESS);
 space_register_ = getSpaceRegister(); assert(space_register_);
 tensor_op_factory_ = TensorOpFactory::get();
 scopes_.push(std::pair<std::string,ScopeId>{"GLOBAL",0}); //GLOBAL scope 0 is automatically open (top scope)
 tensor_rt_->openScope("GLOBAL");
}
#else
NumServer::NumServer(const std::string & graph_executor_name,
                     const std::string & node_executor_name):
 contr_seq_optimizer_("dummy"),
 tensor_rt_(std::make_shared<runtime::TensorRuntime>(graph_executor_name,node_executor_name))
{
 num_processes_ = 1; process_rank_ = 0;
 space_register_ = getSpaceRegister(); assert(space_register_);
 tensor_op_factory_ = TensorOpFactory::get();
 scopes_.push(std::pair<std::string,ScopeId>{"GLOBAL",0}); //GLOBAL scope 0 is automatically open (top scope)
 tensor_rt_->openScope("GLOBAL");
}
#endif


NumServer::~NumServer()
{
 destroyOrphanedTensors();
 auto iter = tensors_.begin();
 while(iter != tensors_.end()){
  std::shared_ptr<TensorOperation> destroy_op = tensor_op_factory_->createTensorOp(TensorOpCode::DESTROY);
  destroy_op->setTensorOperand(iter->second);
  auto submitted = submit(destroy_op);
  if(submitted) submitted = sync(*destroy_op);
  iter = tensors_.begin();
 }
 tensor_rt_->closeScope();
 scopes_.pop();
}


#ifdef MPI_ENABLED
void NumServer::reconfigureTensorRuntime(const MPICommProxy & communicator,
                                         const std::string & dag_executor_name,
                                         const std::string & node_executor_name)
{
 bool synced = tensor_rt_->sync(); assert(synced);
 tensor_rt_ = std::move(std::make_shared<runtime::TensorRuntime>(communicator,dag_executor_name,node_executor_name));
 return;
}
#else
void NumServer::reconfigureTensorRuntime(const std::string & dag_executor_name,
                                         const std::string & node_executor_name)
{
 bool synced = tensor_rt_->sync(); assert(synced);
 tensor_rt_ = std::move(std::make_shared<runtime::TensorRuntime>(dag_executor_name,node_executor_name));
 return;
}
#endif

void NumServer::resetContrSeqOptimizer(const std::string & optimizer_name)
{
 contr_seq_optimizer_ = optimizer_name;
 return;
}

void NumServer::resetRuntimeLoggingLevel(int level)
{
 if(tensor_rt_) tensor_rt_->resetLoggingLevel(level);
 return;
}

void NumServer::registerTensorMethod(const std::string & tag, std::shared_ptr<TensorMethod> method)
{
 auto res = ext_methods_.insert({tag,method});
 if(!(std::get<1>(res))) std::cout << "#ERROR(NumServer::registerTensorMethod): Method already exists: " << tag << std::endl;
 assert(std::get<1>(res));
 return;
}

std::shared_ptr<TensorMethod> NumServer::getTensorMethod(const std::string & tag)
{
 return ext_methods_[tag];
}

void NumServer::registerExternalData(const std::string & tag, std::shared_ptr<BytePacket> packet)
{
 auto res = ext_data_.insert({tag,packet});
 if(!(std::get<1>(res))) std::cout << "#ERROR(NumServer::registerExternalData): Data already exists: " << tag << std::endl;
 assert(std::get<1>(res));
 return;
}

std::shared_ptr<BytePacket> NumServer::getExternalData(const std::string & tag)
{
 return ext_data_[tag];
}


ScopeId NumServer::openScope(const std::string & scope_name)
{
 assert(scope_name.length() > 0);
 ScopeId new_scope_id = scopes_.size();
 scopes_.push(std::pair<std::string,ScopeId>{scope_name,new_scope_id});
 return new_scope_id;
}

ScopeId NumServer::closeScope()
{
 assert(!scopes_.empty());
 const auto & prev_scope = scopes_.top();
 ScopeId prev_scope_id = std::get<1>(prev_scope);
 scopes_.pop();
 return prev_scope_id;
}


SpaceId NumServer::createVectorSpace(const std::string & space_name, DimExtent space_dim,
                                     const VectorSpace ** space_ptr)
{
 assert(space_name.length() > 0);
 SpaceId space_id = space_register_->registerSpace(std::make_shared<VectorSpace>(space_dim,space_name));
 if(space_ptr != nullptr) *space_ptr = space_register_->getSpace(space_id);
 return space_id;
}

void NumServer::destroyVectorSpace(const std::string & space_name)
{
 assert(false);
 //`Finish
 return;
}

void NumServer::destroyVectorSpace(SpaceId space_id)
{
 assert(false);
 //`Finish
 return;
}

const VectorSpace * NumServer::getVectorSpace(const std::string & space_name) const
{
 return space_register_->getSpace(space_name);
}


SubspaceId NumServer::createSubspace(const std::string & subspace_name,
                                     const std::string & space_name,
                                     std::pair<DimOffset,DimOffset> bounds,
                                     const Subspace ** subspace_ptr)
{
 assert(subspace_name.length() > 0 && space_name.length() > 0);
 const VectorSpace * space = space_register_->getSpace(space_name);
 assert(space != nullptr);
 SubspaceId subspace_id = space_register_->registerSubspace(std::make_shared<Subspace>(space,bounds,subspace_name));
 if(subspace_ptr != nullptr) *subspace_ptr = space_register_->getSubspace(space_name,subspace_name);
 auto res = subname2id_.insert({subspace_name,space->getRegisteredId()});
 if(!(res.second)) std::cout << "#ERROR(NumServer::createSubspace): Subspace already exists: " << subspace_name << std::endl;
 assert(res.second);
 return subspace_id;
}

void NumServer::destroySubspace(const std::string & subspace_name)
{
 assert(false);
 //`Finish
 return;
}

void NumServer::destroySubspace(SubspaceId subspace_id)
{
 assert(false);
 //`Finish
 return;
}

const Subspace * NumServer::getSubspace(const std::string & subspace_name) const
{
 assert(subspace_name.length() > 0);
 auto it = subname2id_.find(subspace_name);
 if(it == subname2id_.end()) std::cout << "#ERROR(NumServer::getSubspace): Subspace not found: " << subspace_name << std::endl;
 assert(it != subname2id_.end());
 SpaceId space_id = (*it).second;
 const VectorSpace * space = space_register_->getSpace(space_id);
 assert(space != nullptr);
 const std::string & space_name = space->getName();
 assert(space_name.length() > 0);
 return space_register_->getSubspace(space_name,subspace_name);
}

bool NumServer::submit(std::shared_ptr<TensorOperation> operation)
{
 bool submitted = false;
 if(operation){
  submitted = true;
  if(operation->getOpcode() == TensorOpCode::CREATE){ //TENSOR_CREATE sets tensor element type for future references
   auto tensor = operation->getTensorOperand(0);
   auto elem_type = std::dynamic_pointer_cast<numerics::TensorOpCreate>(operation)->getTensorElementType();
   if(elem_type == TensorElementType::VOID){
    elem_type = tensor->getElementType();
    std::dynamic_pointer_cast<numerics::TensorOpCreate>(operation)->resetTensorElementType(elem_type);
   }else{
    tensor->setElementType(elem_type);
   }
   auto res = tensors_.emplace(std::make_pair(tensor->getName(),tensor));
   if(!(res.second)){
    std::cout << "#ERROR(exatn::NumServer::submit): Attempt to CREATE an already existing tensor "
              << tensor->getName() << std::endl;
    submitted = false;
   }
  }else if(operation->getOpcode() == TensorOpCode::DESTROY){
   auto tensor = operation->getTensorOperand(0);
   auto num_deleted = tensors_.erase(tensor->getName());
   if(num_deleted != 1){
    std::cout << "#ERROR(exatn::NumServer::submit): Attempt to DESTROY a non-existing tensor "
              << tensor->getName() << std::endl;
    submitted = false;
   }
  }
  if(submitted) tensor_rt_->submit(operation);
 }
 return submitted;
}

bool NumServer::submit(TensorNetwork & network)
{
 std::vector<unsigned int> process_set(num_processes_);
 for(unsigned int i = 0; i < num_processes_; ++i) process_set[i] = i;
 return submit(network,process_set);
}

bool NumServer::submit(std::shared_ptr<TensorNetwork> network)
{
 std::vector<unsigned int> process_set(num_processes_);
 for(unsigned int i = 0; i < num_processes_; ++i) process_set[i] = i;
 return submit(network,process_set);
}

bool NumServer::submit(TensorNetwork & network,
                       const std::vector<unsigned int> & process_set)
{
 assert(network.isValid()); //debug
 auto & op_list = network.getOperationList(contr_seq_optimizer_);
 auto output_tensor = network.getTensor(0);
 auto iter = tensors_.find(output_tensor->getName());
 bool submitted = false;
 if(iter == tensors_.end()){ //output tensor does not exist and needs to be created
  implicit_tensors_.emplace_back(output_tensor); //list of implicitly created tensors (for garbage collection)
  //Create output tensor:
  std::shared_ptr<TensorOperation> op0 = tensor_op_factory_->createTensorOp(TensorOpCode::CREATE);
  op0->setTensorOperand(output_tensor);
  std::dynamic_pointer_cast<numerics::TensorOpCreate>(op0)->
   resetTensorElementType(output_tensor->getElementType());
  submitted = submit(op0); if(!submitted) return false; //this CREATE operation will also register the output tensor
 }
 //Initialize output tensor to zero:
 std::shared_ptr<TensorOperation> op1 = tensor_op_factory_->createTensorOp(TensorOpCode::TRANSFORM);
 op1->setTensorOperand(output_tensor);
 std::dynamic_pointer_cast<numerics::TensorOpTransform>(op1)->
  resetFunctor(std::shared_ptr<TensorMethod>(new numerics::FunctorInitVal(0.0)));
 submitted = submit(op1); if(!submitted) return false;
 //Submit all tensor operations for tensor network evaluation:
 for(auto op = op_list.begin(); op != op_list.end(); ++op){
  submitted = submit(*op); if(!submitted) return false;
 }
 return true;
}

bool NumServer::submit(std::shared_ptr<TensorNetwork> network,
                       const std::vector<unsigned int> & process_set)
{
 if(network) return submit(*network,process_set);
 return false;
}

bool NumServer::submit(TensorExpansion & expansion,
                       std::shared_ptr<Tensor> accumulator)
{
 std::vector<unsigned int> process_set(num_processes_);
 for(unsigned int i = 0; i < num_processes_; ++i) process_set[i] = i;
 return submit(expansion,accumulator,process_set);
}

bool NumServer::submit(std::shared_ptr<TensorExpansion> expansion,
                       std::shared_ptr<Tensor> accumulator)
{
 std::vector<unsigned int> process_set(num_processes_);
 for(unsigned int i = 0; i < num_processes_; ++i) process_set[i] = i;
 return submit(expansion,accumulator,process_set);
}

bool NumServer::submit(TensorExpansion & expansion,
                       std::shared_ptr<Tensor> accumulator,
                       const std::vector<unsigned int> & process_set)
{
 assert(accumulator);
 std::list<std::shared_ptr<TensorOperation>> accumulations;
 for(auto component = expansion.begin(); component != expansion.end(); ++component){
  //Evaluate the tensor network component (compute its output tensor):
  auto & network = *(component->network_);
  auto submitted = submit(network); if(!submitted) return false;
  //Create accumulation operation for the scaled computed output tensor:
  bool conjugated;
  auto output_tensor = network.getTensor(0,&conjugated); assert(!conjugated); //output tensor cannot be conjugated
  std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::ADD);
  op->setTensorOperand(accumulator);
  op->setTensorOperand(output_tensor,conjugated);
  op->setScalar(0,component->coefficient_);
  std::string add_pattern;
  auto generated = generate_addition_pattern(accumulator->getRank(),add_pattern); assert(generated);
  op->setIndexPattern(add_pattern);
  accumulations.emplace_back(op);
 }
 //Submit all previously created accumulation operations:
 for(auto & accumulation: accumulations){
  auto submitted = submit(accumulation); if(!submitted) return false;
 }
 return true;
}

bool NumServer::submit(std::shared_ptr<TensorExpansion> expansion,
                       std::shared_ptr<Tensor> accumulator,
                       const std::vector<unsigned int> & process_set)
{
 if(expansion) return submit(*expansion,accumulator,process_set);
 return false;
}

bool NumServer::sync(const Tensor & tensor, bool wait)
{
 return tensor_rt_->sync(tensor,wait);
}

bool NumServer::sync(TensorOperation & operation, bool wait)
{
 return tensor_rt_->sync(operation,wait);
}

bool NumServer::sync(TensorNetwork & network, bool wait)
{
 return sync(*(network.getTensor(0)),wait);
}

bool NumServer::sync(bool wait)
{
 return tensor_rt_->sync(wait);
}

bool NumServer::sync(const std::string & name, bool wait)
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::sync): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 return sync(*(iter->second),wait);
}

std::shared_ptr<Tensor> NumServer::getTensor(const std::string & name)
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::getTensor): Tensor " << name << " not found!" << std::endl;
  assert(false);
 }
 return iter->second;
}

Tensor & NumServer::getTensorRef(const std::string & name)
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::getTensorRef): Tensor " << name << " not found!" << std::endl;
  assert(false);
 }
 return *(iter->second);
}

TensorElementType NumServer::getTensorElementType(const std::string & name) const
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::getTensorElementType): Tensor " << name << " not found!" << std::endl;
  assert(false);
 }
 return (iter->second)->getElementType();
}

bool NumServer::registerTensorIsometry(const std::string & name,
                                       const std::vector<unsigned int> & iso_dims)
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::registerTensorIsometry): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 iter->second->registerIsometry(iso_dims);
 return true;
}

bool NumServer::registerTensorIsometry(const std::string & name,
                                       const std::vector<unsigned int> & iso_dims0,
                                       const std::vector<unsigned int> & iso_dims1)
{
 auto registered = registerTensorIsometry(name,iso_dims0);
 if(registered) registered = registerTensorIsometry(name,iso_dims1);
 return registered;
}

bool NumServer::createTensor(std::shared_ptr<Tensor> tensor,
                             TensorElementType element_type)
{
 assert(tensor);
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CREATE);
 op->setTensorOperand(tensor);
 std::dynamic_pointer_cast<numerics::TensorOpCreate>(op)->resetTensorElementType(element_type);
 auto submitted = submit(op);
 return submitted;
}

bool NumServer::createTensorSync(std::shared_ptr<Tensor> tensor,
                                 TensorElementType element_type)
{
 assert(tensor);
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CREATE);
 op->setTensorOperand(tensor);
 std::dynamic_pointer_cast<numerics::TensorOpCreate>(op)->resetTensorElementType(element_type);
 auto submitted = submit(op);
 if(submitted) submitted = sync(*op);
 return submitted;
}

bool NumServer::destroyTensor(const std::string & name) //always synchronous
{
 destroyOrphanedTensors(); //garbage collection
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::destroyTensor): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::DESTROY);
 op->setTensorOperand(iter->second);
 auto submitted = submit(op);
 if(submitted) submitted = sync(*op);
 return submitted;
}

bool NumServer::destroyTensorSync(const std::string & name)
{
 destroyOrphanedTensors(); //garbage collection
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::destroyTensorSync): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::DESTROY);
 op->setTensorOperand(iter->second);
 auto submitted = submit(op);
 if(submitted) submitted = sync(*op);
 return submitted;
}

bool NumServer::initTensorRnd(const std::string & name)
{
 return transformTensor(name,std::shared_ptr<TensorMethod>(new numerics::FunctorInitRnd()));
}

bool NumServer::initTensorRndSync(const std::string & name)
{
 return transformTensorSync(name,std::shared_ptr<TensorMethod>(new numerics::FunctorInitRnd()));
}

bool NumServer::transformTensor(const std::string & name, std::shared_ptr<TensorMethod> functor)
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::transformTensor): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::TRANSFORM);
 op->setTensorOperand(iter->second);
 std::dynamic_pointer_cast<numerics::TensorOpTransform>(op)->resetFunctor(functor);
 auto submitted = submit(op);
 return submitted;
}

bool NumServer::transformTensorSync(const std::string & name, std::shared_ptr<TensorMethod> functor)
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::transformTensorSync): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::TRANSFORM);
 op->setTensorOperand(iter->second);
 std::dynamic_pointer_cast<numerics::TensorOpTransform>(op)->resetFunctor(functor);
 auto submitted = submit(op);
 if(submitted) submitted = sync(*op);
 return submitted;
}

bool NumServer::decomposeTensorSVD(const std::string & contraction)
{
 //`Implement
 return false;
}

bool NumServer::decomposeTensorSVDSync(const std::string & contraction)
{
 //`Implement
 return false;
}

bool NumServer::decomposeTensorSVDL(const std::string & contraction)
{
 //`Implement
 return false;
}

bool NumServer::decomposeTensorSVDLSync(const std::string & contraction)
{
 //`Implement
 return false;
}

bool NumServer::decomposeTensorSVDR(const std::string & contraction)
{
 //`Implement
 return false;
}

bool NumServer::decomposeTensorSVDRSync(const std::string & contraction)
{
 //`Implement
 return false;
}

bool NumServer::decomposeTensorSVDLR(const std::string & contraction)
{
 //`Implement
 return false;
}

bool NumServer::decomposeTensorSVDLRSync(const std::string & contraction)
{
 //`Implement
 return false;
}

bool NumServer::orthogonalizeTensorSVD(const std::string & contraction)
{
 //`Implement
 return false;
}

bool NumServer::orthogonalizeTensorSVDSync(const std::string & contraction)
{
 //`Implement
 return false;
}

bool NumServer::orthogonalizeTensorMGS(const std::string & name)
{
 //`Implement
 return false;
}

bool NumServer::orthogonalizeTensorMGSSync(const std::string & name)
{
 //`Implement
 return false;
}

bool NumServer::evaluateTensorNetwork(const std::string & name, const std::string & network)
{
 std::vector<std::string> tensors;
 auto parsed = parse_tensor_network(network,tensors);
 if(parsed){
  std::map<std::string,std::shared_ptr<Tensor>> tensor_map;
  std::string tensor_name;
  std::vector<IndexLabel> indices;
  for(const auto & tensor: tensors){
   bool complex_conj;
   parsed = parse_tensor(tensor,tensor_name,indices,complex_conj);
   if(!parsed){
    std::cout << "#ERROR(exatn::NumServer::evaluateTensorNetwork): Invalid tensor: " << tensor << std::endl;
    std::cout << "#ERROR(exatn::NumServer::evaluateTensorNetwork): Invalid tensor network: " << network << std::endl;
    break;
   }
   auto iter = tensors_.find(tensor_name);
   if(iter == tensors_.end()){
    std::cout << "#ERROR(exatn::NumServer::evaluateTensorNetwork): Tensor " << tensor_name << " not found!" << std::endl;
    std::cout << "#ERROR(exatn::NumServer::evaluateTensorNetwork): Undefined tensor in tensor network: " << network << std::endl;
    parsed = false;
    break;
   }
   auto res = tensor_map.emplace(std::make_pair(tensor_name,iter->second));
   parsed = res.second; if(!parsed) break;
  }
  if(parsed){
   TensorNetwork tensnet(name,network,tensor_map);
   parsed = submit(tensnet);
  }
 }else{
  std::cout << "#ERROR(exatn::NumServer::evaluateTensorNetwork): Invalid tensor network: " << network << std::endl;
 }
 return parsed;
}

bool NumServer::evaluateTensorNetworkSync(const std::string & name, const std::string & network)
{
 std::vector<std::string> tensors;
 auto parsed = parse_tensor_network(network,tensors);
 if(parsed){
  std::map<std::string,std::shared_ptr<Tensor>> tensor_map;
  std::string tensor_name;
  std::vector<IndexLabel> indices;
  for(const auto & tensor: tensors){
   bool complex_conj;
   parsed = parse_tensor(tensor,tensor_name,indices,complex_conj);
   if(!parsed){
    std::cout << "#ERROR(exatn::NumServer::evaluateTensorNetworkSync): Invalid tensor: " << tensor << std::endl;
    std::cout << "#ERROR(exatn::NumServer::evaluateTensorNetworkSync): Invalid tensor network: " << network << std::endl;
    break;
   }
   auto iter = tensors_.find(tensor_name);
   if(iter == tensors_.end()){
    std::cout << "#ERROR(exatn::NumServer::evaluateTensorNetworkSync): Tensor " << tensor_name << " not found!" << std::endl;
    std::cout << "#ERROR(exatn::NumServer::evaluateTensorNetworkSync): Undefined tensor in tensor network: " << network << std::endl;
    parsed = false;
    break;
   }
   auto res = tensor_map.emplace(std::make_pair(tensor_name,iter->second));
   parsed = res.second; if(!parsed) break;
  }
  if(parsed){
   TensorNetwork tensnet(name,network,tensor_map);
   parsed = submit(tensnet);
   if(parsed) parsed = sync(tensnet);
  }
 }else{
  std::cout << "#ERROR(exatn::NumServer::evaluateTensorNetworkSync): Invalid tensor network: " << network << std::endl;
 }
 return parsed;
}

std::shared_ptr<talsh::Tensor> NumServer::getLocalTensor(std::shared_ptr<Tensor> tensor, //in: exatn::numerics::Tensor to get slice of (by copy)
                         const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec) //in: tensor slice specification
{
 return (tensor_rt_->getLocalTensor(tensor,slice_spec)).get();
}

std::shared_ptr<talsh::Tensor> NumServer::getLocalTensor(std::shared_ptr<Tensor> tensor) //in: exatn::numerics::Tensor to get slice of (by copy)
{
 const auto tensor_rank = tensor->getRank();
 std::vector<std::pair<DimOffset,DimExtent>> slice_spec(tensor_rank);
 for(unsigned int i = 0; i < tensor_rank; ++i) slice_spec[i] = std::pair<DimOffset,DimExtent>{0,tensor->getDimExtent(i)};
 return getLocalTensor(tensor,slice_spec);
}

std::shared_ptr<talsh::Tensor> NumServer::getLocalTensor(const std::string & name,
                   const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec)
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()) return std::shared_ptr<talsh::Tensor>(nullptr);
 return getLocalTensor(iter->second,slice_spec);
}

std::shared_ptr<talsh::Tensor> NumServer::getLocalTensor(const std::string & name)
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()) return std::shared_ptr<talsh::Tensor>(nullptr);
 return getLocalTensor(iter->second);
}

void NumServer::destroyOrphanedTensors()
{
 auto iter = implicit_tensors_.begin();
 while(iter != implicit_tensors_.end()){
  if(iter->unique()){
   std::shared_ptr<TensorOperation> destroy_op = tensor_op_factory_->createTensorOp(TensorOpCode::DESTROY);
   destroy_op->setTensorOperand(*iter);
   auto submitted = submit(destroy_op);
   iter = implicit_tensors_.erase(iter);
  }else{
   ++iter;
  }
 }
 return;
}

} //namespace exatn
