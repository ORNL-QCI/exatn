/** ExaTN::Numerics: Numerical server
REVISION: 2021/03/02

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "num_server.hpp"
#include "tensor_range.hpp"
#include "timers.hpp"

#include <complex>
#include <vector>
#include <stack>
#include <list>
#include <map>
#include <future>
#include <algorithm>

#ifdef MPI_ENABLED
#include "mpi.h"
#endif

#include <cstddef>

namespace exatn{

/** Numerical server (singleton) **/
std::shared_ptr<NumServer> numericalServer {nullptr}; //initialized by exatn::initialize()


#ifdef MPI_ENABLED
NumServer::NumServer(const MPICommProxy & communicator,
                     const ParamConf & parameters,
                     const std::string & graph_executor_name,
                     const std::string & node_executor_name):
 contr_seq_optimizer_("metis"), contr_seq_caching_(false), logging_(0), intra_comm_(communicator)
{
 int mpi_error = MPI_Comm_size(*(communicator.get<MPI_Comm>()),&num_processes_); assert(mpi_error == MPI_SUCCESS);
 mpi_error = MPI_Comm_rank(*(communicator.get<MPI_Comm>()),&process_rank_); assert(mpi_error == MPI_SUCCESS);
 mpi_error = MPI_Comm_rank(MPI_COMM_WORLD,&global_process_rank_); assert(mpi_error == MPI_SUCCESS);
 process_world_ = std::make_shared<ProcessGroup>(intra_comm_,num_processes_);
 std::vector<unsigned int> myself = {static_cast<unsigned int>(process_rank_)};
 process_self_ = std::make_shared<ProcessGroup>(MPICommProxy(MPI_COMM_SELF),myself);
 int64_t provided_buf_size; //bytes
 if(parameters.getParameter("host_memory_buffer_size",&provided_buf_size)){
  process_world_->resetMemoryLimitPerProcess(static_cast<std::size_t>(provided_buf_size));
  process_self_->resetMemoryLimitPerProcess(static_cast<std::size_t>(provided_buf_size));
 }
 initBytePacket(&byte_packet_);
 space_register_ = getSpaceRegister(); assert(space_register_);
 tensor_op_factory_ = TensorOpFactory::get();
 mpi_error = MPI_Barrier(*(communicator.get<MPI_Comm>())); assert(mpi_error == MPI_SUCCESS);
 time_start_ = exatn::Timer::timeInSecHR();
 tensor_rt_ = std::move(std::make_shared<runtime::TensorRuntime>(communicator,parameters,graph_executor_name,node_executor_name));
 scopes_.push(std::pair<std::string,ScopeId>{"GLOBAL",0}); //GLOBAL scope 0 is automatically open (top scope)
 tensor_rt_->openScope("GLOBAL");
}
#else
NumServer::NumServer(const ParamConf & parameters,
                     const std::string & graph_executor_name,
                     const std::string & node_executor_name):
 contr_seq_optimizer_("metis"), contr_seq_caching_(false), logging_(0)
{
 num_processes_ = 1; process_rank_ = 0; global_process_rank_ = 0;
 process_world_ = std::make_shared<ProcessGroup>(intra_comm_,num_processes_); //intra-communicator is empty here
 std::vector<unsigned int> myself = {static_cast<unsigned int>(process_rank_)};
 process_self_ = std::make_shared<ProcessGroup>(intra_comm_,myself); //intra-communicator is empty here
 int64_t provided_buf_size; //bytes
 if(parameters.getParameter("host_memory_buffer_size",&provided_buf_size)){
  process_world_->resetMemoryLimitPerProcess(static_cast<std::size_t>(provided_buf_size));
  process_self_->resetMemoryLimitPerProcess(static_cast<std::size_t>(provided_buf_size));
 }
 initBytePacket(&byte_packet_);
 space_register_ = getSpaceRegister(); assert(space_register_);
 tensor_op_factory_ = TensorOpFactory::get();
 time_start_ = exatn::Timer::timeInSecHR();
 tensor_rt_ = std::move(std::make_shared<runtime::TensorRuntime>(parameters,graph_executor_name,node_executor_name));
 scopes_.push(std::pair<std::string,ScopeId>{"GLOBAL",0}); //GLOBAL scope 0 is automatically open (top scope)
 tensor_rt_->openScope("GLOBAL");
}
#endif


NumServer::~NumServer()
{
 destroyOrphanedTensors(); //garbage collection
 auto iter = tensors_.begin();
 while(iter != tensors_.end()){
  std::shared_ptr<TensorOperation> destroy_op = tensor_op_factory_->createTensorOp(TensorOpCode::DESTROY);
  destroy_op->setTensorOperand(iter->second);
  auto submitted = submit(destroy_op);
  if(submitted) submitted = sync(*destroy_op);
  iter = tensors_.begin();
 }
 tensor_rt_->closeScope(); //contains sync() inside
 scopes_.pop();
 destroyBytePacket(&byte_packet_);
 resetClientLoggingLevel();
}


#ifdef MPI_ENABLED
void NumServer::reconfigureTensorRuntime(const MPICommProxy & communicator,
                                         const ParamConf & parameters,
                                         const std::string & dag_executor_name,
                                         const std::string & node_executor_name)
{
 while(!tensor_rt_);
 bool synced = tensor_rt_->sync(); assert(synced);
 tensor_rt_ = std::move(std::make_shared<runtime::TensorRuntime>(communicator,parameters,dag_executor_name,node_executor_name));
 return;
}
#else
void NumServer::reconfigureTensorRuntime(const ParamConf & parameters,
                                         const std::string & dag_executor_name,
                                         const std::string & node_executor_name)
{
 while(!tensor_rt_);
 bool synced = tensor_rt_->sync(); assert(synced);
 tensor_rt_ = std::move(std::make_shared<runtime::TensorRuntime>(parameters,dag_executor_name,node_executor_name));
 return;
}
#endif

void NumServer::resetContrSeqOptimizer(const std::string & optimizer_name, bool caching)
{
 contr_seq_optimizer_ = optimizer_name;
 contr_seq_caching_ = caching;
 return;
}

void NumServer::activateContrSeqCaching()
{
 contr_seq_caching_ = true;
 return;
}

void NumServer::deactivateContrSeqCaching()
{
 contr_seq_caching_ = false;
 return;
}

bool NumServer::queryContrSeqCaching() const
{
 return contr_seq_caching_;
}

void NumServer::resetClientLoggingLevel(int level){
 if(logging_ == 0){
  if(level != 0) logfile_.open("exatn_main_thread."+std::to_string(global_process_rank_)+".log", std::ios::out | std::ios::trunc);
 }else{
  if(level == 0) logfile_.close();
 }
 logging_ = level;
 return;
}

void NumServer::resetRuntimeLoggingLevel(int level)
{
 while(!tensor_rt_);
 bool synced = tensor_rt_->sync(); assert(synced);
 tensor_rt_->resetLoggingLevel(level);
 return;
}

void NumServer::activateFastMath()
{
 while(!tensor_rt_);
 bool synced = tensor_rt_->sync(); assert(synced);
 tensor_rt_->activateFastMath();
 if(logging_ > 0){
  logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
           << "]: Fast math activated (if available); Tensor runtime synced" << std::endl << std::flush;
 }
 synced = tensor_rt_->sync(); assert(synced);
 return;
}

std::size_t NumServer::getMemoryBufferSize() const
{
 while(!tensor_rt_);
 return tensor_rt_->getMemoryBufferSize();
}

double NumServer::getTotalFlopCount() const
{
 while(!tensor_rt_);
 return tensor_rt_->getTotalFlopCount();
}

const ProcessGroup & NumServer::getDefaultProcessGroup() const
{
 return *process_world_;
}

const ProcessGroup & NumServer::getCurrentProcessGroup() const
{
 return *process_self_;
}

int NumServer::getProcessRank() const
{
 return process_rank_;
}

int NumServer::getNumProcesses() const
{
 return num_processes_;
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

bool NumServer::submitOp(std::shared_ptr<TensorOperation> operation)
{
 bool submitted = false;
 if(operation){
  if(logging_ > 1){
   logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
            << "]: Submitting tensor operation:" << std::endl;
   operation->printItFile(logfile_);
   logfile_.flush(); //`Debug only
  }
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
              << tensor->getName() << std::endl << std::flush;
    submitted = false;
   }
  }
  if(submitted) tensor_rt_->submit(operation);
 }
 return submitted;
}

bool NumServer::submit(std::shared_ptr<TensorOperation> operation)
{
 bool success = true;
 const auto num_operands = operation->getNumOperands();
 auto elem_type = TensorElementType::VOID;
 for(unsigned int i = 0; i < num_operands; ++i){
  auto operand = operation->getTensorOperand(i);
  elem_type = operand->getElementType();
  if(elem_type != TensorElementType::VOID) break;
 }
 std::stack<unsigned int> deltas;
 for(unsigned int i = 0; i < num_operands; ++i){
  auto operand = operation->getTensorOperand(i);
  const auto & tensor_name = operand->getName();
  if(tensor_name.length() >= 2){
   if(tensor_name[0] == '_' && tensor_name[1] == 'd'){ //_d: explicit Kronecker Delta tensor
    assert(elem_type != TensorElementType::VOID);
    std::shared_ptr<TensorOperation> op0 = tensor_op_factory_->createTensorOp(TensorOpCode::CREATE);
    op0->setTensorOperand(operand);
    std::dynamic_pointer_cast<numerics::TensorOpCreate>(op0)->resetTensorElementType(elem_type);
    success = submitOp(op0);
    if(success){
     deltas.push(i);
     std::shared_ptr<TensorOperation> op1 = tensor_op_factory_->createTensorOp(TensorOpCode::TRANSFORM);
     op1->setTensorOperand(operand);
     std::dynamic_pointer_cast<numerics::TensorOpTransform>(op1)->
      resetFunctor(std::shared_ptr<TensorMethod>(new numerics::FunctorInitDelta()));
     success = submitOp(op1);
    }
   }
  }
  if(!success) break;
 }
 if(success){
  success = submitOp(operation);
  if(success){
   while(!deltas.empty()){
    auto operand = operation->getTensorOperand(deltas.top());
    std::shared_ptr<TensorOperation> op2 = tensor_op_factory_->createTensorOp(TensorOpCode::DESTROY);
    op2->setTensorOperand(operand);
    success = submitOp(op2); if(!success) break;
    deltas.pop();
   }
  }
 }
 return success;
}

bool NumServer::submit(TensorNetwork & network)
{
 return submit(getDefaultProcessGroup(),network);
}

bool NumServer::submit(std::shared_ptr<TensorNetwork> network)
{
 return submit(getDefaultProcessGroup(),network);
}

bool NumServer::submit(const ProcessGroup & process_group,
                       TensorNetwork & network)
{
 const bool debugging = false;
 const bool serialize = false;

 //Determine parallel execution configuration:
 unsigned int local_rank; //local process rank within the process group
 if(!process_group.rankIsIn(process_rank_,&local_rank)) return true; //process is not in the group: Do nothing
 assert(network.isValid()); //debug
 unsigned int num_procs = process_group.getSize(); //number of executing processes
 assert(local_rank < num_procs);
 if(logging_ > 0) logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
                           << "]: Submitting tensor network <" << network.getName() << "> (" << network.getTensor(0)->getName()
                           << ") for execution by " << num_procs << " processes with memory limit "
                           << process_group.getMemoryLimitPerProcess() << " bytes" << std::endl << std::flush;
 if(logging_ > 0) network.printItFile(logfile_);

 //Determine the pseudo-optimal tensor contraction sequence:
 const auto num_input_tensors = network.getNumTensors();
 bool new_contr_seq = network.exportContractionSequence().empty();
 if(contr_seq_caching_ && new_contr_seq){ //check whether the optimal tensor contraction sequence is already available from the past
  auto cached_seq = ContractionSeqOptimizer::findContractionSequence(network);
  if(cached_seq.first != nullptr){
   network.importContractionSequence(*(cached_seq.first),cached_seq.second);
   new_contr_seq = false;
  }
 }
 if(new_contr_seq){
  double flops = network.determineContractionSequence(contr_seq_optimizer_);
 }
 if(logging_ > 0) logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
                           << "]: Found a contraction sequence candidate locally (caching = " << contr_seq_caching_
                           << ")" << std::endl;

#ifdef MPI_ENABLED
 //Synchronize on the best tensor contraction sequence across processes:
 if(num_procs > 1 && num_input_tensors > 2){
  double flops = 0.0;
  std::vector<double> proc_flops(num_procs,0.0);
  std::vector<unsigned int> contr_seq_content;
  packContractionSequenceIntoVector(network.exportContractionSequence(&flops),contr_seq_content);
  auto errc = MPI_Gather(&flops,1,MPI_DOUBLE,proc_flops.data(),1,MPI_DOUBLE,
                         0,process_group.getMPICommProxy().getRef<MPI_Comm>());
  assert(errc == MPI_SUCCESS);
  auto min_flops = std::min_element(proc_flops.cbegin(),proc_flops.cend());
  int root_id = static_cast<int>(std::distance(proc_flops.cbegin(),min_flops));
  errc = MPI_Bcast(&root_id,1,MPI_INT,0,process_group.getMPICommProxy().getRef<MPI_Comm>());
  assert(errc == MPI_SUCCESS);
  errc = MPI_Bcast(&flops,1,MPI_DOUBLE,root_id,process_group.getMPICommProxy().getRef<MPI_Comm>());
  assert(errc == MPI_SUCCESS);
  errc = MPI_Bcast(contr_seq_content.data(),contr_seq_content.size(),MPI_UNSIGNED,
                   root_id,process_group.getMPICommProxy().getRef<MPI_Comm>());
  assert(errc == MPI_SUCCESS);
  network.importContractionSequence(contr_seq_content,flops);
 }
 if(logging_ > 0) logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
                           << "]: Found the optimal contraction sequence across all processes" << std::endl;
#endif

 //Generate the primitive tensor operation list:
 auto & op_list = network.getOperationList(contr_seq_optimizer_,(num_procs > 1));
 if(contr_seq_caching_ && new_contr_seq) ContractionSeqOptimizer::cacheContractionSequence(network);
 const double max_intermediate_presence_volume = network.getMaxIntermediatePresenceVolume();
 unsigned int max_intermediate_rank = 0;
 double max_intermediate_volume = network.getMaxIntermediateVolume(&max_intermediate_rank);
 if(logging_ > 0) logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
                           << "]: Contraction info: FMA flop count = " << std::scientific << network.getFMAFlops()
                           << "; Max intermediate presence volume = " << max_intermediate_presence_volume
                           << "; Max intermediate rank = " << max_intermediate_rank
                           << " with volume " << max_intermediate_volume << " -> ";

 //Split some of the tensor network indices based on the requested memory limit:
 const std::size_t proc_mem_volume = process_group.getMemoryLimitPerProcess() / sizeof(std::complex<double>);
 if(max_intermediate_presence_volume > 0.0 && max_intermediate_volume > 0.0){
  const double shrink_coef = std::min(1.0,
   static_cast<double>(proc_mem_volume) / (max_intermediate_presence_volume * 1.5 * 2.0)); //{1.5:memory fragmentation}; {2.0:tensor transpose}
  max_intermediate_volume *= shrink_coef;
 }
 if(logging_ > 0) logfile_ << max_intermediate_volume << " (after slicing)" << std::endl << std::flush;
 //if(max_intermediate_presence_volume > 0.0 && max_intermediate_volume > 0.0)
 network.splitIndices(static_cast<std::size_t>(max_intermediate_volume));
 if(logging_ > 0) network.printSplitIndexInfo(logfile_,logging_ > 1);

 //Create the output tensor of the tensor network if needed:
 bool submitted = false;
 auto output_tensor = network.getTensor(0);
 auto iter = tensors_.find(output_tensor->getName());
 if(iter == tensors_.end()){ //output tensor does not exist and needs to be created
  implicit_tensors_.emplace_back(output_tensor); //list of implicitly created tensors (for garbage collection)
  //Create output tensor:
  std::shared_ptr<TensorOperation> op0 = tensor_op_factory_->createTensorOp(TensorOpCode::CREATE);
  op0->setTensorOperand(output_tensor);
  std::dynamic_pointer_cast<numerics::TensorOpCreate>(op0)->
   resetTensorElementType(output_tensor->getElementType());
  submitted = submit(op0); if(!submitted) return false; //this CREATE operation will also register the output tensor
 }

 //Initialize the output tensor to zero:
 std::shared_ptr<TensorOperation> op1 = tensor_op_factory_->createTensorOp(TensorOpCode::TRANSFORM);
 op1->setTensorOperand(output_tensor);
 std::dynamic_pointer_cast<numerics::TensorOpTransform>(op1)->
  resetFunctor(std::shared_ptr<TensorMethod>(new numerics::FunctorInitVal(0.0)));
 submitted = submit(op1); if(!submitted) return false;
 //Submit all tensor operations for tensor network evaluation:
 const auto num_split_indices = network.getNumSplitIndices(); //total number of indices that were split
 if(logging_ > 0) logfile_ << "Number of split indices = " << num_split_indices << std::endl << std::flush;
 std::size_t num_items_executed = 0; //number of tensor sub-networks executed
 if(num_split_indices > 0){ //multiple tensor sub-networks need to be executed by all processes ditributively
  //Distribute tensor sub-networks among processes:
  std::vector<DimExtent> work_extents(num_split_indices);
  for(int i = 0; i < num_split_indices; ++i) work_extents[i] = network.getSplitIndexInfo(i).second.size(); //number of segments per split index
  numerics::TensorRange work_range(work_extents); //each range dimension refers to the number of segments per the corresponding split index
  bool not_done = true;
  if(num_procs > 1) not_done = work_range.reset(num_procs,local_rank); //work subrange for the current local process rank (may be empty)
  if(logging_ > 0) logfile_ << "Total number of sub-networks = " << work_range.localVolume()
                            << "; Current process has a share (0/1) = " << not_done << std::endl << std::flush;
  //Each process executes its share of tensor sub-networks:
  while(not_done){
   if(logging_ > 1){
    logfile_ << "Submitting sub-network ";
    work_range.printCurrent(logfile_);
    logfile_ << std::endl;
   }
   std::unordered_map<numerics::TensorHashType,std::shared_ptr<numerics::Tensor>> intermediate_slices; //temporary slices of intermediates
   std::list<std::shared_ptr<numerics::Tensor>> input_slices; //temporary slices of input tensors
   //Execute all tensor operations for the current tensor sub-network:
   for(auto op = op_list.begin(); op != op_list.end(); ++op){
    if(debugging && logging_ > 1){ //debug
     logfile_ << "Next tensor operation from the tensor network operation list:" << std::endl;
     (*op)->printItFile(logfile_);
    }
    const auto num_operands = (*op)->getNumOperands();
    std::shared_ptr<TensorOperation> tens_op = (*op)->clone();
    //Substitute sliced tensor operands with their respective slices from the current tensor sub-network:
    std::shared_ptr<numerics::Tensor> output_tensor_slice;
    for(unsigned int op_num = 0; op_num < num_operands; ++op_num){
     auto tensor = (*op)->getTensorOperand(op_num);
     const auto tensor_rank = tensor->getRank();
     if(debugging && logging_ > 1){ //debug
      logfile_ << "Next tensor operand " << op_num << " named " << tensor->getName();
     }
     bool tensor_is_output;
     bool tensor_is_intermediate = tensorNameIsIntermediate(*tensor,&tensor_is_output);
     tensor_is_output = (tensor == output_tensor);
     //Look up the tensor operand in the table of sliced tensor operands:
     std::pair<numerics::TensorHashType,numerics::TensorHashType> key;
     if(tensor_is_intermediate || tensor_is_output){ //intermediate tensor (including output tensor)
      numerics::TensorHashType zero = 0;
      key = std::make_pair(zero,tensor->getTensorHash());
     }else{ //input tensor
      numerics::TensorHashType pos = op_num;
      key = std::make_pair((*op)->getTensorOpHash(),pos);
     }
     const auto * tensor_info = network.getSplitTensorInfo(key);
     //Replace the full tensor operand with its respective slice (if found):
     if(tensor_info != nullptr){ //tensor has splitted indices
      if(debugging && logging_ > 1) logfile_ << " with split indices" << std::endl; //debug
      std::shared_ptr<numerics::Tensor> tensor_slice;
      //Look up the tensor slice in case it has already been created:
      if(tensor_is_intermediate && (!tensor_is_output)){ //pure intermediate tensor
       auto slice_iter = intermediate_slices.find(tensor->getTensorHash()); //look up by the hash of the parental tensor
       if(slice_iter != intermediate_slices.end()) tensor_slice = slice_iter->second;
      }
      //Create the tensor slice upon first encounter:
      if(!tensor_slice){
       //Import original subspaces and dimension extents from the parental tensor:
       std::vector<SubspaceId> subspaces(tensor_rank);
       for(unsigned int i = 0; i < tensor_rank; ++i) subspaces[i] = tensor->getDimSubspaceId(i);
       std::vector<DimExtent> dim_extents(tensor_rank);
       for(unsigned int i = 0; i < tensor_rank; ++i) dim_extents[i] = tensor->getDimExtent(i);
       //Replace the sliced dimensions with their updated subspaces and dimension extents:
       for(const auto & index_desc: *tensor_info){
        const auto gl_index_id = index_desc.first;
        const auto index_pos = index_desc.second;
        const auto & index_info = network.getSplitIndexInfo(gl_index_id);
        const auto segment_selector = work_range.getIndex(gl_index_id);
        subspaces[index_pos] = index_info.second[segment_selector].first;
        dim_extents[index_pos] = index_info.second[segment_selector].second;
        if(logging_ > 1) logfile_ << "Index replacement in tensor " << tensor->getName()
         << ": " << index_info.first << " in position " << index_pos << std::endl;
       }
       //Construct the tensor slice from the parental tensor:
       tensor_slice = tensor->createSubtensor(subspaces,dim_extents);
       tensor_slice->rename(); //unique automatic name will be generated
       //Store the tensor in the table for subsequent referencing:
       if(tensor_is_intermediate && (!tensor_is_output)){ //pure intermediate tensor
        auto res = intermediate_slices.emplace(std::make_pair(tensor->getTensorHash(),tensor_slice));
        assert(res.second);
       }else{ //input/output tensor
        input_slices.emplace_back(tensor_slice);
       }
      }
      //Replace the sliced tensor operand with its current slice in the primary tensor operation:
      bool replaced = tens_op->resetTensorOperand(op_num,tensor_slice); assert(replaced);
      //Allocate the input/output tensor slice and extract its contents (not for intermediates):
      if(!tensor_is_intermediate || tensor_is_output){ //input/output tensor: create slice and extract its contents
       //Create an empty slice of the input/output tensor:
       std::shared_ptr<TensorOperation> create_slice = tensor_op_factory_->createTensorOp(TensorOpCode::CREATE);
       create_slice->setTensorOperand(tensor_slice);
       std::dynamic_pointer_cast<numerics::TensorOpCreate>(create_slice)->
        resetTensorElementType(tensor->getElementType());
       submitted = submit(create_slice); if(!submitted) return false;
       //Extract the slice contents from the input/output tensor:
       if(tensor_is_output){ //make sure the output tensor slice only shows up once
        //assert(tensor == output_tensor);
        assert(!output_tensor_slice);
        output_tensor_slice = tensor_slice;
       }
       std::shared_ptr<TensorOperation> extract_slice = tensor_op_factory_->createTensorOp(TensorOpCode::SLICE);
       extract_slice->setTensorOperand(tensor_slice);
       extract_slice->setTensorOperand(tensor);
       submitted = submit(extract_slice); if(!submitted) return false;
      }
     }else{
      if(debugging && logging_ > 1) logfile_ << " without split indices" << std::endl; //debug
     }
    } //loop over tensor operands
    //Submit the primary tensor operation with the current slices:
    submitted = submit(tens_op); if(!submitted) return false;
    //Insert the output tensor slice back into the output tensor:
    if(output_tensor_slice){
     std::shared_ptr<TensorOperation> insert_slice = tensor_op_factory_->createTensorOp(TensorOpCode::INSERT);
     insert_slice->setTensorOperand(output_tensor);
     insert_slice->setTensorOperand(output_tensor_slice);
     submitted = submit(insert_slice); if(!submitted) return false;
     output_tensor_slice.reset();
    }
    //Destroy temporary input tensor slices:
    for(auto & input_slice: input_slices){
     std::shared_ptr<TensorOperation> destroy_slice = tensor_op_factory_->createTensorOp(TensorOpCode::DESTROY);
     destroy_slice->setTensorOperand(input_slice);
     submitted = submit(destroy_slice); if(!submitted) return false;
    }
    if(serialize) sync(process_group); //sync for serialization
    input_slices.clear();
   } //loop over tensor operations
   //Erase intermediate tensor slices once all tensor operations have been executed:
   intermediate_slices.clear();
   ++num_items_executed;
   //Proceed to the next tensor sub-network:
   not_done = work_range.next();
  } //loop over tensor sub-networks
  //Allreduce the tensor network output tensor within the executing process group:
  if(num_procs > 1){
   std::shared_ptr<TensorOperation> allreduce = tensor_op_factory_->createTensorOp(TensorOpCode::ALLREDUCE);
   allreduce->setTensorOperand(output_tensor);
   std::dynamic_pointer_cast<numerics::TensorOpAllreduce>(allreduce)->resetMPICommunicator(process_group.getMPICommProxy());
   submitted = submit(allreduce); if(!submitted) return false;
  }
 }else{ //only a single tensor (sub-)network executed redundantly by all processes
  for(auto op = op_list.begin(); op != op_list.end(); ++op){
   submitted = submit(*op); if(!submitted) return false;
  }
  ++num_items_executed;
 }
 if(logging_ > 0) logfile_ << "Number of submitted sub-networks = " << num_items_executed << std::endl << std::flush;
 return true;
}

bool NumServer::submit(const ProcessGroup & process_group,
                       std::shared_ptr<TensorNetwork> network)
{
 if(network) return submit(process_group,*network);
 return false;
}

bool NumServer::submit(TensorExpansion & expansion,
                       std::shared_ptr<Tensor> accumulator)
{
 return submit(getDefaultProcessGroup(),expansion,accumulator);
}

bool NumServer::submit(std::shared_ptr<TensorExpansion> expansion,
                       std::shared_ptr<Tensor> accumulator)
{
 return submit(getDefaultProcessGroup(),expansion,accumulator);
}

bool NumServer::submit(const ProcessGroup & process_group,
                       TensorExpansion & expansion,
                       std::shared_ptr<Tensor> accumulator)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
 assert(accumulator);
 std::list<std::shared_ptr<TensorOperation>> accumulations;
 for(auto component = expansion.begin(); component != expansion.end(); ++component){
  //Evaluate the tensor network component (compute its output tensor):
  auto & network = *(component->network);
  auto submitted = submit(process_group,network); if(!submitted) return false;
  //Create accumulation operation for the scaled computed output tensor:
  bool conjugated;
  auto output_tensor = network.getTensor(0,&conjugated); assert(!conjugated); //output tensor cannot be conjugated
  std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::ADD);
  op->setTensorOperand(accumulator);
  op->setTensorOperand(output_tensor,conjugated);
  op->setScalar(0,component->coefficient);
  std::string add_pattern;
  auto generated = generate_addition_pattern(accumulator->getRank(),add_pattern,false,
                                             accumulator->getName(),output_tensor->getName());
  assert(generated);
  op->setIndexPattern(add_pattern);
  accumulations.emplace_back(op);
 }
 //Submit all previously created accumulation operations:
 for(auto & accumulation: accumulations){
  auto submitted = submit(accumulation); if(!submitted) return false;
 }
 return true;
}

bool NumServer::submit(const ProcessGroup & process_group,
                       std::shared_ptr<TensorExpansion> expansion,
                       std::shared_ptr<Tensor> accumulator)
{
 if(expansion) return submit(process_group,*expansion,accumulator);
 return false;
}

bool NumServer::sync(const Tensor & tensor, bool wait)
{
 return sync(getCurrentProcessGroup(),tensor,wait);
}

bool NumServer::sync(const ProcessGroup & process_group, const Tensor & tensor, bool wait)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
 auto success = tensor_rt_->sync(tensor,wait);
 if(success){
  if(logging_ > 0) logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
   << "]: Locally synchronized all operations on tensor <" << tensor.getName() << ">" << std::endl << std::flush;
#ifdef MPI_ENABLED
  auto errc = MPI_Barrier(process_group.getMPICommProxy().getRef<MPI_Comm>());
  success = success && (errc == MPI_SUCCESS);
  if(logging_ > 0) logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
   << "]: Globally synchronized all operations on tensor <" << tensor.getName() << ">" << std::endl << std::flush;
#endif
 }
 return success;
}

bool NumServer::sync(TensorOperation & operation, bool wait) //`Local synchronization semantics
{
 return tensor_rt_->sync(operation,wait);
}

bool NumServer::sync(TensorNetwork & network, bool wait)
{
 return sync(getCurrentProcessGroup(),network,wait);
}

bool NumServer::sync(const ProcessGroup & process_group, TensorNetwork & network, bool wait)
{
 return sync(process_group,*(network.getTensor(0)),wait); //synchronization on the output tensor of the tensor network
}

bool NumServer::sync(bool wait)
{
 return sync(getCurrentProcessGroup(),wait);
}

bool NumServer::sync(const ProcessGroup & process_group, bool wait)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
 destroyOrphanedTensors(); //garbage collection
 auto success = tensor_rt_->sync(wait);
 if(success){
  if(logging_ > 0) logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
   << "]: Locally synchronized all operations" << std::endl << std::flush;
#ifdef MPI_ENABLED
  auto errc = MPI_Barrier(process_group.getMPICommProxy().getRef<MPI_Comm>());
  success = success && (errc == MPI_SUCCESS);
  if(logging_ > 0) logfile_ << "[" << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(getTimeStampStart())
   << "]: Globally synchronized all operations" << std::endl << std::flush;
#endif
 }
 return success;
}

bool NumServer::sync(const std::string & name, bool wait)
{
 return sync(getCurrentProcessGroup(),name,wait);
}

bool NumServer::sync(const ProcessGroup & process_group, const std::string & name, bool wait)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::sync): Tensor " << name << " not found!" << std::endl << std::flush;
  assert(false);
 }
 return sync(process_group,*(iter->second),wait);
}

bool NumServer::tensorAllocated(const std::string & name) const
{
 return (tensors_.find(name) != tensors_.cend());
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

bool NumServer::createTensor(const std::string & name,
                             const TensorSignature & signature,
                             TensorElementType element_type)
{
 const unsigned int tensor_rank = signature.getRank();
 std::vector<DimExtent> extents(tensor_rank);
 for(unsigned int i = 0; i < tensor_rank; ++i){
  const auto * subspace = space_register_->getSubspace(signature.getDimSpaceId(i),signature.getDimSubspaceId(i));
  if(subspace != nullptr){
   extents[i] = subspace->getDimension();
  }else{
   std::cout << "#ERROR(exatn::NumServer::createTensor): Unregistered subspace passed!" << std::endl << std::flush;
   assert(false);
  }
 }
 return createTensor(name,element_type,TensorShape(extents),signature);
}

bool NumServer::createTensor(std::shared_ptr<Tensor> tensor,
                             TensorElementType element_type)
{
 return createTensor(getDefaultProcessGroup(),tensor,element_type);
}

bool NumServer::createTensorSync(std::shared_ptr<Tensor> tensor,
                                 TensorElementType element_type)
{
 return createTensorSync(getDefaultProcessGroup(),tensor,element_type);
}

bool NumServer::createTensor(const ProcessGroup & process_group,
                             std::shared_ptr<Tensor> tensor,
                             TensorElementType element_type)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
 assert(tensor);
 bool submitted = false;
 if(element_type != TensorElementType::VOID){
  std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CREATE);
  op->setTensorOperand(tensor);
  std::dynamic_pointer_cast<numerics::TensorOpCreate>(op)->resetTensorElementType(element_type);
  submitted = submit(op);
 }else{
  std::cout << "#ERROR(exatn::createTensor): Missing data type!" << std::endl;
 }
 return submitted;
}

bool NumServer::createTensorSync(const ProcessGroup & process_group,
                                 std::shared_ptr<Tensor> tensor,
                                 TensorElementType element_type)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
 assert(tensor);
 bool submitted = false;
 if(element_type != TensorElementType::VOID){
  std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CREATE);
  op->setTensorOperand(tensor);
  std::dynamic_pointer_cast<numerics::TensorOpCreate>(op)->resetTensorElementType(element_type);
  submitted = submit(op);
  if(submitted) submitted = sync(*op);
 }else{
  std::cout << "#ERROR(exatn::createTensor): Missing data type!" << std::endl;
 }
 return submitted;
}

bool NumServer::createTensors(TensorNetwork & tensor_network,
                              TensorElementType element_type)
{
 return createTensors(getDefaultProcessGroup(),tensor_network,element_type);
}

bool NumServer::createTensorsSync(TensorNetwork & tensor_network,
                                  TensorElementType element_type)
{
 return createTensorsSync(getDefaultProcessGroup(),tensor_network,element_type);
}

bool NumServer::createTensors(const ProcessGroup & process_group,
                              TensorNetwork & tensor_network,
                              TensorElementType element_type)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
 bool success = true;
 for(auto tens = tensor_network.begin(); tens != tensor_network.end(); ++tens){
  auto tensor = tens->second.getTensor();
  const auto & tens_name = tensor->getName();
  if(!tensorAllocated(tens_name)) success = createTensor(process_group,tensor,element_type);
  if(!success) break;
 }
 return success;
}

bool NumServer::createTensorsSync(const ProcessGroup & process_group,
                                  TensorNetwork & tensor_network,
                                  TensorElementType element_type)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
 bool success = true;
 for(auto tens = tensor_network.begin(); tens != tensor_network.end(); ++tens){
  auto tensor = tens->second.getTensor();
  const auto & tens_name = tensor->getName();
  if(!tensorAllocated(tens_name)) success = createTensorSync(process_group,tensor,element_type);
  if(!success) break;
 }
 return success;
}

bool NumServer::destroyTensor(const std::string & name) //always synchronous
{
 destroyOrphanedTensors(); //garbage collection
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#WARNING(exatn::NumServer::destroyTensor): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::DESTROY);
 op->setTensorOperand(iter->second);
 auto submitted = submit(op);
 if(submitted) submitted = sync(*op); //`Do I need to always sync Destroy?
 return submitted;
}

bool NumServer::destroyTensorSync(const std::string & name)
{
 destroyOrphanedTensors(); //garbage collection
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#WARNING(exatn::NumServer::destroyTensorSync): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::DESTROY);
 op->setTensorOperand(iter->second);
 auto submitted = submit(op);
 if(submitted) submitted = sync(*op);
 return submitted;
}

bool NumServer::destroyTensors(TensorNetwork & tensor_network)
{
 bool success = true;
 for(auto tens = tensor_network.begin(); tens != tensor_network.end(); ++tens){
  auto tensor = tens->second.getTensor();
  const auto & tens_name = tensor->getName();
  if(tensorAllocated(tens_name)) success = destroyTensor(tens_name);
  if(!success) break;
 }
 return success;
}

bool NumServer::destroyTensorsSync(TensorNetwork & tensor_network)
{
 bool success = true;
 for(auto tens = tensor_network.begin(); tens != tensor_network.end(); ++tens){
  auto tensor = tens->second.getTensor();
  const auto & tens_name = tensor->getName();
  if(tensorAllocated(tens_name)) success = destroyTensorSync(tens_name);
  if(!success) break;
 }
 return success;
}

bool NumServer::initTensorFile(const std::string & name,
                               const std::string & filename)
{
 return transformTensor(name,std::shared_ptr<TensorMethod>(new numerics::FunctorInitFile(filename)));
}

bool NumServer::initTensorFileSync(const std::string & name,
                                   const std::string & filename)
{
 return transformTensorSync(name,std::shared_ptr<TensorMethod>(new numerics::FunctorInitFile(filename)));
}

bool NumServer::initTensorRnd(const std::string & name)
{
 return transformTensor(name,std::shared_ptr<TensorMethod>(new numerics::FunctorInitRnd()));
}

bool NumServer::initTensorRndSync(const std::string & name)
{
 return transformTensorSync(name,std::shared_ptr<TensorMethod>(new numerics::FunctorInitRnd()));
}

bool NumServer::initTensorsRnd(TensorNetwork & tensor_network)
{
 bool success = true;
 for(auto tens = tensor_network.begin(); tens != tensor_network.end(); ++tens){
  auto tensor = tens->second.getTensor();
  const auto & tens_name = tensor->getName();
  if(tensorAllocated(tens_name)){
   if(tens->first == 0){ //output tensor
    success = initTensor(tens_name,0.0);
   }else{ //input tensor
    success = initTensorRnd(tens_name);
   }
  }else{
   success = false;
  }
  if(!success) break;
 }
 return success;
}

bool NumServer::initTensorsRndSync(TensorNetwork & tensor_network)
{
 bool success = true;
 for(auto tens = tensor_network.begin(); tens != tensor_network.end(); ++tens){
  auto tensor = tens->second.getTensor();
  const auto & tens_name = tensor->getName();
  if(tensorAllocated(tens_name)){
   if(tens->first == 0){ //output tensor
    success = initTensor(tens_name,0.0);
   }else{ //input tensor
    success = initTensorRnd(tens_name);
   }
  }else{
   success = false;
  }
  if(!success) break;
 }
 return success;
}

bool NumServer::computeMaxAbsSync(const std::string & name,
                                  double & norm)
{
 norm = -1.0;
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::computeMaxAbsSync): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 auto functor = std::shared_ptr<TensorMethod>(new numerics::FunctorMaxAbs());
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::TRANSFORM);
 op->setTensorOperand(iter->second);
 std::dynamic_pointer_cast<numerics::TensorOpTransform>(op)->resetFunctor(functor);
 auto submitted = submit(op);
 if(submitted){
  submitted = sync(*op);
  if(submitted) norm = std::dynamic_pointer_cast<numerics::FunctorMaxAbs>(functor)->getNorm();
 }
 return submitted;
}

bool NumServer::computeNorm1Sync(const std::string & name,
                                 double & norm)
{
 norm = -1.0;
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::computeNorm1Sync): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 auto functor = std::shared_ptr<TensorMethod>(new numerics::FunctorNorm1());
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::TRANSFORM);
 op->setTensorOperand(iter->second);
 std::dynamic_pointer_cast<numerics::TensorOpTransform>(op)->resetFunctor(functor);
 auto submitted = submit(op);
 if(submitted){
  submitted = sync(*op);
  if(submitted) norm = std::dynamic_pointer_cast<numerics::FunctorNorm1>(functor)->getNorm();
 }
 return submitted;
}

bool NumServer::computeNorm2Sync(const std::string & name,
                                 double & norm)
{
 norm = -1.0;
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::computeNorm2Sync): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 auto functor = std::shared_ptr<TensorMethod>(new numerics::FunctorNorm2());
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::TRANSFORM);
 op->setTensorOperand(iter->second);
 std::dynamic_pointer_cast<numerics::TensorOpTransform>(op)->resetFunctor(functor);
 auto submitted = submit(op);
 if(submitted){
  submitted = sync(*op);
  if(submitted) norm = std::dynamic_pointer_cast<numerics::FunctorNorm2>(functor)->getNorm();
 }
 return submitted;
}

bool NumServer::computePartialNormsSync(const std::string & name,            //in: tensor name
                                        unsigned int tensor_dimension,       //in: chosen tensor dimension
                                        std::vector<double> & partial_norms) //out: partial 2-norms over the chosen tensor dimension
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::computePartialNormsSync): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 if(tensor_dimension >= iter->second->getRank()){
  std::cout << "#ERROR(exatn::NumServer::computePartialNormsSync): Chosen tensor dimension " << tensor_dimension
            << " does not exist for tensor " << name << std::endl;
  return false;
 }
 auto functor = std::shared_ptr<TensorMethod>(new numerics::FunctorDiagRank(tensor_dimension));
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::TRANSFORM);
 op->setTensorOperand(iter->second);
 std::dynamic_pointer_cast<numerics::TensorOpTransform>(op)->resetFunctor(functor);
 auto submitted = submit(op);
 if(submitted){
  submitted = sync(*op);
  if(submitted){
   const auto & norms = std::dynamic_pointer_cast<numerics::FunctorDiagRank>(functor)->getPartialNorms();
   submitted = !norms.empty();
   if(submitted) partial_norms.assign(norms.cbegin(),norms.cend());
  }
 }
 return submitted;
}

bool NumServer::replicateTensor(const std::string & name, int root_process_rank)
{
 return replicateTensor(getDefaultProcessGroup(),name,root_process_rank);
}

bool NumServer::replicateTensorSync(const std::string & name, int root_process_rank)
{
 return replicateTensorSync(getDefaultProcessGroup(),name,root_process_rank);
}

bool NumServer::replicateTensor(const ProcessGroup & process_group, const std::string & name, int root_process_rank)
{
 unsigned int local_rank; //local process rank within the process group
 if(!process_group.rankIsIn(process_rank_,&local_rank)) return true; //process is not in the group: Do nothing
 auto iter = tensors_.find(name);
 //Broadcast the tensor meta-data:
 int byte_packet_len = 0;
 if(local_rank == root_process_rank){
  if(iter != tensors_.end()){
   iter->second->pack(byte_packet_);
   byte_packet_len = static_cast<int>(byte_packet_.size_bytes); assert(byte_packet_len > 0);
  }else{
   std::cout << "#ERROR(exatn::NumServer::replicateTensor): Tensor " << name << " not found at root!" << std::endl;
   assert(false);
  }
 }
#ifdef MPI_ENABLED
 auto errc = MPI_Bcast(&byte_packet_len,1,MPI_INT,root_process_rank,
                       process_group.getMPICommProxy().getRef<MPI_Comm>());
 assert(errc == MPI_SUCCESS);
 if(local_rank != root_process_rank) byte_packet_.size_bytes = byte_packet_len;
 errc = MPI_Bcast(byte_packet_.base_addr,byte_packet_len,MPI_UNSIGNED_CHAR,root_process_rank,
                  process_group.getMPICommProxy().getRef<MPI_Comm>());
 assert(errc == MPI_SUCCESS);
#endif
 //Create the tensor locally if it did not exist:
 resetBytePacket(&byte_packet_);
 if(iter == tensors_.end()){ //only other MPI processes than root_process_rank
  auto tensor = std::make_shared<Tensor>(byte_packet_);
  std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CREATE);
  op->setTensorOperand(tensor);
  std::dynamic_pointer_cast<numerics::TensorOpCreate>(op)->resetTensorElementType(tensor->getElementType());
  auto submitted = submit(op);
  if(submitted) submitted = sync(*op);
  assert(submitted);
 }
 clearBytePacket(&byte_packet_);
 //Broadcast the tensor body:
#ifdef MPI_ENABLED
// errc = MPI_Barrier(process_group.getMPICommProxy().getRef<MPI_Comm>()); assert(errc == MPI_SUCCESS);
#endif
 return broadcastTensor(process_group,name,root_process_rank);
}

bool NumServer::replicateTensorSync(const ProcessGroup & process_group, const std::string & name, int root_process_rank)
{
 unsigned int local_rank; //local process rank within the process group
 if(!process_group.rankIsIn(process_rank_,&local_rank)) return true; //process is not in the group: Do nothing
 auto iter = tensors_.find(name);
 //Broadcast the tensor meta-data:
 int byte_packet_len = 0;
 if(local_rank == root_process_rank){
  if(iter != tensors_.end()){
   iter->second->pack(byte_packet_);
   byte_packet_len = static_cast<int>(byte_packet_.size_bytes); assert(byte_packet_len > 0);
  }else{
   std::cout << "#ERROR(exatn::NumServer::replicateTensorSync): Tensor " << name << " not found at root!" << std::endl;
   assert(false);
  }
 }
#ifdef MPI_ENABLED
 auto errc = MPI_Bcast(&byte_packet_len,1,MPI_INT,root_process_rank,
                       process_group.getMPICommProxy().getRef<MPI_Comm>());
 assert(errc == MPI_SUCCESS);
 if(local_rank != root_process_rank) byte_packet_.size_bytes = byte_packet_len;
 errc = MPI_Bcast(byte_packet_.base_addr,byte_packet_len,MPI_UNSIGNED_CHAR,root_process_rank,
                  process_group.getMPICommProxy().getRef<MPI_Comm>());
 assert(errc == MPI_SUCCESS);
#endif
 //Create the tensor locally if it did not exist:
 resetBytePacket(&byte_packet_);
 if(iter == tensors_.end()){ //only other MPI processes than root_process_rank
  auto tensor = std::make_shared<Tensor>(byte_packet_);
  std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CREATE);
  op->setTensorOperand(tensor);
  std::dynamic_pointer_cast<numerics::TensorOpCreate>(op)->resetTensorElementType(tensor->getElementType());
  auto submitted = submit(op);
  if(submitted) submitted = sync(*op);
  assert(submitted);
 }
 clearBytePacket(&byte_packet_);
 //Broadcast the tensor body:
#ifdef MPI_ENABLED
// errc = MPI_Barrier(process_group.getMPICommProxy().getRef<MPI_Comm>()); assert(errc == MPI_SUCCESS);
#endif
 return broadcastTensorSync(process_group,name,root_process_rank);
}

bool NumServer::broadcastTensor(const std::string & name, int root_process_rank)
{
 return broadcastTensor(getDefaultProcessGroup(),name,root_process_rank);
}

bool NumServer::broadcastTensorSync(const std::string & name, int root_process_rank)
{
 return broadcastTensorSync(getDefaultProcessGroup(),name,root_process_rank);
}

bool NumServer::broadcastTensor(const ProcessGroup & process_group, const std::string & name, int root_process_rank)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::broadcastTensor): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::BROADCAST);
 op->setTensorOperand(iter->second);
 std::dynamic_pointer_cast<numerics::TensorOpBroadcast>(op)->resetMPICommunicator(process_group.getMPICommProxy());
 std::dynamic_pointer_cast<numerics::TensorOpBroadcast>(op)->resetRootRank(root_process_rank);
 auto submitted = submit(op);
 return submitted;
}

bool NumServer::broadcastTensorSync(const ProcessGroup & process_group, const std::string & name, int root_process_rank)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::broadcastTensorSync): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::BROADCAST);
 op->setTensorOperand(iter->second);
 std::dynamic_pointer_cast<numerics::TensorOpBroadcast>(op)->resetMPICommunicator(process_group.getMPICommProxy());
 std::dynamic_pointer_cast<numerics::TensorOpBroadcast>(op)->resetRootRank(root_process_rank);
 auto submitted = submit(op);
 if(submitted) submitted = sync(*op);
 return submitted;
}

bool NumServer::allreduceTensor(const std::string & name)
{
 return allreduceTensor(getDefaultProcessGroup(),name);
}

bool NumServer::allreduceTensorSync(const std::string & name)
{
 return allreduceTensorSync(getDefaultProcessGroup(),name);
}

bool NumServer::allreduceTensor(const ProcessGroup & process_group, const std::string & name)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::allreduceTensor): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::ALLREDUCE);
 op->setTensorOperand(iter->second);
 std::dynamic_pointer_cast<numerics::TensorOpAllreduce>(op)->resetMPICommunicator(process_group.getMPICommProxy());
 auto submitted = submit(op);
 return submitted;
}

bool NumServer::allreduceTensorSync(const ProcessGroup & process_group, const std::string & name)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::allreduceTensorSync): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::ALLREDUCE);
 op->setTensorOperand(iter->second);
 std::dynamic_pointer_cast<numerics::TensorOpAllreduce>(op)->resetMPICommunicator(process_group.getMPICommProxy());
 auto submitted = submit(op);
 if(submitted) submitted = sync(*op);
 return submitted;
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

bool NumServer::transformTensor(const std::string & name, const std::string & functor_name)
{
 return transformTensor(name,getTensorMethod(functor_name));
}

bool NumServer::transformTensorSync(const std::string & name, const std::string & functor_name)
{
 return transformTensorSync(name,getTensorMethod(functor_name));
}

bool NumServer::extractTensorSlice(const std::string & tensor_name,
                                   const std::string & slice_name)
{
 bool success = false;
 auto iter = tensors_.find(tensor_name);
 if(iter != tensors_.end()){
  auto tensor0 = iter->second;
  iter = tensors_.find(slice_name);
  if(iter != tensors_.end()){
   auto tensor1 = iter->second;
   std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::SLICE);
   op->setTensorOperand(tensor1);
   op->setTensorOperand(tensor0);
   success = submit(op);
  }else{
   success = false;
   std::cout << "#ERROR(exatn::NumServer::extractTensorSlice): Tensor " << slice_name << " not found!\n";
  }
 }else{
  success = false;
  std::cout << "#ERROR(exatn::NumServer::extractTensorSlice): Tensor " << tensor_name << " not found!\n";
 }
 return success;
}

bool NumServer::extractTensorSliceSync(const std::string & tensor_name,
                                       const std::string & slice_name)
{
 bool success = false;
 auto iter = tensors_.find(tensor_name);
 if(iter != tensors_.end()){
  auto tensor0 = iter->second;
  iter = tensors_.find(slice_name);
  if(iter != tensors_.end()){
   auto tensor1 = iter->second;
   std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::SLICE);
   op->setTensorOperand(tensor1);
   op->setTensorOperand(tensor0);
   success = submit(op);
   if(success) success = sync(*op);
  }else{
   success = false;
   std::cout << "#ERROR(exatn::NumServer::extractTensorSliceSync): Tensor " << slice_name << " not found!\n";
  }
 }else{
  success = false;
  std::cout << "#ERROR(exatn::NumServer::extractTensorSliceSync): Tensor " << tensor_name << " not found!\n";
 }
 return success;
}

bool NumServer::insertTensorSlice(const std::string & tensor_name,
                                  const std::string & slice_name)
{
 bool success = false;
 auto iter = tensors_.find(tensor_name);
 if(iter != tensors_.end()){
  auto tensor0 = iter->second;
  iter = tensors_.find(slice_name);
  if(iter != tensors_.end()){
   auto tensor1 = iter->second;
   std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::INSERT);
   op->setTensorOperand(tensor0);
   op->setTensorOperand(tensor1);
   success = submit(op);
  }else{
   success = false;
   std::cout << "#ERROR(exatn::NumServer::insertTensorSlice): Tensor " << slice_name << " not found!\n";
  }
 }else{
  success = false;
  std::cout << "#ERROR(exatn::NumServer::insertTensorSlice): Tensor " << tensor_name << " not found!\n";
 }
 return success;
}

bool NumServer::insertTensorSliceSync(const std::string & tensor_name,
                                      const std::string & slice_name)
{
 bool success = false;
 auto iter = tensors_.find(tensor_name);
 if(iter != tensors_.end()){
  auto tensor0 = iter->second;
  iter = tensors_.find(slice_name);
  if(iter != tensors_.end()){
   auto tensor1 = iter->second;
   std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::INSERT);
   op->setTensorOperand(tensor0);
   op->setTensorOperand(tensor1);
   success = submit(op);
   if(success) success = sync(*op);
  }else{
   success = false;
   std::cout << "#ERROR(exatn::NumServer::insertTensorSliceSync): Tensor " << slice_name << " not found!\n";
  }
 }else{
  success = false;
  std::cout << "#ERROR(exatn::NumServer::insertTensorSliceSync): Tensor " << tensor_name << " not found!\n";
 }
 return success;
}

bool NumServer::decomposeTensorSVD(const std::string & contraction)
{
 std::vector<std::string> tensors;
 auto parsed = parse_tensor_network(contraction,tensors);
 if(parsed){
  if(tensors.size() == 4){
   std::string tensor_name;
   std::vector<IndexLabel> indices;
   bool complex_conj0,complex_conj1,complex_conj2,complex_conj3;
   parsed = parse_tensor(tensors[0],tensor_name,indices,complex_conj0);
   if(parsed){
    assert(!complex_conj0);
    auto iter = tensors_.find(tensor_name);
    if(iter != tensors_.end()){
     auto tensor0 = iter->second;
     parsed = parse_tensor(tensors[1],tensor_name,indices,complex_conj1);
     if(parsed){
      assert(!complex_conj1);
      iter = tensors_.find(tensor_name);
      if(iter != tensors_.end()){
       auto tensor1 = iter->second;
       parsed = parse_tensor(tensors[2],tensor_name,indices,complex_conj2);
       if(parsed){
        assert(!complex_conj2);
        iter = tensors_.find(tensor_name);
        if(iter != tensors_.end()){
         auto tensor2 = iter->second;
         parsed = parse_tensor(tensors[3],tensor_name,indices,complex_conj3);
         if(parsed){
          assert(!complex_conj3);
          iter = tensors_.find(tensor_name);
          if(iter != tensors_.end()){
           auto tensor3 = iter->second;
           std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::DECOMPOSE_SVD3);
           op->setTensorOperand(tensor1,complex_conj1); //out: left tensor factor
           op->setTensorOperand(tensor3,complex_conj3); //out: right tensor factor
           op->setTensorOperand(tensor2,complex_conj2); //out: middle tensor factor
           op->setTensorOperand(tensor0,complex_conj0); //in: original tensor
           op->setIndexPattern(contraction);
           parsed = sync(*tensor0) && sync(*tensor1) && sync(*tensor2) && sync(*tensor3);
           if(parsed) parsed = submit(op);
          }else{
           parsed = false;
           std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVD): Tensor " << tensor_name << " not found in tensor contraction: "
                     << contraction << std::endl;
          }
         }else{
          std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVD): Invalid argument#3 in tensor contraction: "
                    << contraction << std::endl;
         }
        }else{
         parsed = false;
         std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVD): Tensor " << tensor_name << " not found in tensor contraction: "
                   << contraction << std::endl;
        }
       }else{
        std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVD): Invalid argument#2 in tensor contraction: "
                  << contraction << std::endl;
       }
      }else{
       parsed = false;
       std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVD): Tensor " << tensor_name << " not found in tensor contraction: "
                 << contraction << std::endl;
      }
     }else{
      std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVD): Invalid argument#1 in tensor contraction: "
                << contraction << std::endl;
     }
    }else{
     parsed = false;
     std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVD): Tensor " << tensor_name << " not found in tensor contraction: "
               << contraction << std::endl;
    }
   }else{
    std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVD): Invalid argument#0 in tensor contraction: "
              << contraction << std::endl;
   }
  }else{
   parsed = false;
   std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVD): Invalid number of arguments in tensor contraction: "
             << contraction << std::endl;
  }
 }else{
  std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVD): Invalid tensor contraction: " << contraction << std::endl;
 }
 return parsed;
}

bool NumServer::decomposeTensorSVDSync(const std::string & contraction)
{
 std::vector<std::string> tensors;
 auto parsed = parse_tensor_network(contraction,tensors);
 if(parsed){
  if(tensors.size() == 4){
   std::string tensor_name;
   std::vector<IndexLabel> indices;
   bool complex_conj0,complex_conj1,complex_conj2,complex_conj3;
   parsed = parse_tensor(tensors[0],tensor_name,indices,complex_conj0);
   if(parsed){
    assert(!complex_conj0);
    auto iter = tensors_.find(tensor_name);
    if(iter != tensors_.end()){
     auto tensor0 = iter->second;
     parsed = parse_tensor(tensors[1],tensor_name,indices,complex_conj1);
     if(parsed){
      assert(!complex_conj1);
      iter = tensors_.find(tensor_name);
      if(iter != tensors_.end()){
       auto tensor1 = iter->second;
       parsed = parse_tensor(tensors[2],tensor_name,indices,complex_conj2);
       if(parsed){
        assert(!complex_conj2);
        iter = tensors_.find(tensor_name);
        if(iter != tensors_.end()){
         auto tensor2 = iter->second;
         parsed = parse_tensor(tensors[3],tensor_name,indices,complex_conj3);
         if(parsed){
          assert(!complex_conj3);
          iter = tensors_.find(tensor_name);
          if(iter != tensors_.end()){
           auto tensor3 = iter->second;
           std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::DECOMPOSE_SVD3);
           op->setTensorOperand(tensor1,complex_conj1); //out: left tensor factor
           op->setTensorOperand(tensor3,complex_conj3); //out: right tensor factor
           op->setTensorOperand(tensor2,complex_conj2); //out: middle tensor factor
           op->setTensorOperand(tensor0,complex_conj0); //in: original tensor
           op->setIndexPattern(contraction);
           parsed = sync(*tensor0) && sync(*tensor1) && sync(*tensor2) && sync(*tensor3);
           if(parsed) parsed = submit(op);
           if(parsed) parsed = sync(*op);
          }else{
           parsed = false;
           std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDSync): Tensor " << tensor_name << " not found in tensor contraction: "
                     << contraction << std::endl;
          }
         }else{
          std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDSync): Invalid argument#3 in tensor contraction: "
                    << contraction << std::endl;
         }
        }else{
         parsed = false;
         std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDSync): Tensor " << tensor_name << " not found in tensor contraction: "
                   << contraction << std::endl;
        }
       }else{
        std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDSync): Invalid argument#2 in tensor contraction: "
                  << contraction << std::endl;
       }
      }else{
       parsed = false;
       std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDSync): Tensor " << tensor_name << " not found in tensor contraction: "
                 << contraction << std::endl;
      }
     }else{
      std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDSync): Invalid argument#1 in tensor contraction: "
                << contraction << std::endl;
     }
    }else{
     parsed = false;
     std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDSync): Tensor " << tensor_name << " not found in tensor contraction: "
               << contraction << std::endl;
    }
   }else{
    std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDSync): Invalid argument#0 in tensor contraction: "
              << contraction << std::endl;
   }
  }else{
   parsed = false;
   std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDSync): Invalid number of arguments in tensor contraction: "
             << contraction << std::endl;
  }
 }else{
  std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDSync): Invalid tensor contraction: " << contraction << std::endl;
 }
 return parsed;
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
 std::vector<std::string> tensors;
 auto parsed = parse_tensor_network(contraction,tensors);
 if(parsed){
  if(tensors.size() == 3){
   std::string tensor_name;
   std::vector<IndexLabel> indices;
   bool complex_conj0,complex_conj1,complex_conj2;
   parsed = parse_tensor(tensors[0],tensor_name,indices,complex_conj0);
   if(parsed){
    assert(!complex_conj0);
    auto iter = tensors_.find(tensor_name);
    if(iter != tensors_.end()){
     auto tensor0 = iter->second;
     parsed = parse_tensor(tensors[1],tensor_name,indices,complex_conj1);
     if(parsed){
      assert(!complex_conj1);
      iter = tensors_.find(tensor_name);
      if(iter != tensors_.end()){
       auto tensor1 = iter->second;
       parsed = parse_tensor(tensors[2],tensor_name,indices,complex_conj2);
       if(parsed){
        assert(!complex_conj2);
        iter = tensors_.find(tensor_name);
        if(iter != tensors_.end()){
         auto tensor2 = iter->second;
         std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::DECOMPOSE_SVD2);
         op->setTensorOperand(tensor1,complex_conj1); //out: left tensor factor
         op->setTensorOperand(tensor2,complex_conj2); //out: right tensor factor
         op->setTensorOperand(tensor0,complex_conj0); //in: original tensor
         op->setIndexPattern(contraction);
         parsed = sync(*tensor0) && sync(*tensor1) && sync(*tensor2);
         if(parsed) parsed = submit(op);
        }else{
         parsed = false;
         std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLR): Tensor " << tensor_name << " not found in tensor contraction: "
                   << contraction << std::endl;
        }
       }else{
        std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLR): Invalid argument#2 in tensor contraction: "
                  << contraction << std::endl;
       }
      }else{
       parsed = false;
       std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLR): Tensor " << tensor_name << " not found in tensor contraction: "
                 << contraction << std::endl;
      }
     }else{
      std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLR): Invalid argument#1 in tensor contraction: "
                << contraction << std::endl;
     }
    }else{
     parsed = false;
     std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLR): Tensor " << tensor_name << " not found in tensor contraction: "
               << contraction << std::endl;
    }
   }else{
    std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLR): Invalid argument#0 in tensor contraction: "
              << contraction << std::endl;
   }
  }else{
   parsed = false;
   std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLR): Invalid number of arguments in tensor contraction: "
             << contraction << std::endl;
  }
 }else{
  std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLR): Invalid tensor contraction: " << contraction << std::endl;
 }
 return parsed;
}

bool NumServer::decomposeTensorSVDLRSync(const std::string & contraction)
{
 std::vector<std::string> tensors;
 auto parsed = parse_tensor_network(contraction,tensors);
 if(parsed){
  if(tensors.size() == 3){
   std::string tensor_name;
   std::vector<IndexLabel> indices;
   bool complex_conj0,complex_conj1,complex_conj2;
   parsed = parse_tensor(tensors[0],tensor_name,indices,complex_conj0);
   if(parsed){
    assert(!complex_conj0);
    auto iter = tensors_.find(tensor_name);
    if(iter != tensors_.end()){
     auto tensor0 = iter->second;
     parsed = parse_tensor(tensors[1],tensor_name,indices,complex_conj1);
     if(parsed){
      assert(!complex_conj1);
      iter = tensors_.find(tensor_name);
      if(iter != tensors_.end()){
       auto tensor1 = iter->second;
       parsed = parse_tensor(tensors[2],tensor_name,indices,complex_conj2);
       if(parsed){
        assert(!complex_conj2);
        iter = tensors_.find(tensor_name);
        if(iter != tensors_.end()){
         auto tensor2 = iter->second;
         std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::DECOMPOSE_SVD2);
         op->setTensorOperand(tensor1,complex_conj1); //out: left tensor factor
         op->setTensorOperand(tensor2,complex_conj2); //out: right tensor factor
         op->setTensorOperand(tensor0,complex_conj0); //in: original tensor
         op->setIndexPattern(contraction);
         parsed = sync(*tensor0) && sync(*tensor1) && sync(*tensor2);
         if(parsed) parsed = submit(op);
         if(parsed) parsed = sync(*op);
        }else{
         parsed = false;
         std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLRSync): Tensor " << tensor_name << " not found in tensor contraction: "
                   << contraction << std::endl;
        }
       }else{
        std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLRSync): Invalid argument#2 in tensor contraction: "
                  << contraction << std::endl;
       }
      }else{
       parsed = false;
       std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLRSync): Tensor " << tensor_name << " not found in tensor contraction: "
                 << contraction << std::endl;
      }
     }else{
      std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLRSync): Invalid argument#1 in tensor contraction: "
                << contraction << std::endl;
     }
    }else{
     parsed = false;
     std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLRSync): Tensor " << tensor_name << " not found in tensor contraction: "
               << contraction << std::endl;
    }
   }else{
    std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLRSync): Invalid argument#0 in tensor contraction: "
              << contraction << std::endl;
   }
  }else{
   parsed = false;
   std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLRSync): Invalid number of arguments in tensor contraction: "
             << contraction << std::endl;
  }
 }else{
  std::cout << "#ERROR(exatn::NumServer::decomposeTensorSVDLRSync): Invalid tensor contraction: " << contraction << std::endl;
 }
 return parsed;
}

bool NumServer::orthogonalizeTensorSVD(const std::string & contraction)
{
 std::vector<std::string> tensors;
 auto parsed = parse_tensor_network(contraction,tensors);
 if(parsed){
  if(tensors.size() == 3){
   std::string tensor_name;
   std::vector<IndexLabel> indices;
   bool complex_conj0,complex_conj1,complex_conj2;
   parsed = parse_tensor(tensors[0],tensor_name,indices,complex_conj0);
   if(parsed){
    assert(!complex_conj0);
    auto iter = tensors_.find(tensor_name);
    if(iter != tensors_.end()){
     auto tensor0 = iter->second;
     parsed = parse_tensor(tensors[1],tensor_name,indices,complex_conj1);
     if(parsed){
      assert(!complex_conj1);
      iter = tensors_.find(tensor_name);
      if(iter != tensors_.end()){
       auto tensor1 = iter->second;
       parsed = parse_tensor(tensors[2],tensor_name,indices,complex_conj2);
       if(parsed){
        assert(!complex_conj2);
        iter = tensors_.find(tensor_name);
        if(iter != tensors_.end()){
         auto tensor2 = iter->second;
         std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::ORTHOGONALIZE_SVD);
         op->setTensorOperand(tensor0,complex_conj0);
         op->setIndexPattern(contraction);
         parsed = sync(*tensor0);
         if(parsed) parsed = submit(op);
        }else{
         parsed = false;
         std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVD): Tensor " << tensor_name << " not found in tensor contraction: "
                   << contraction << std::endl;
        }
       }else{
        std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVD): Invalid argument#2 in tensor contraction: "
                  << contraction << std::endl;
       }
      }else{
       parsed = false;
       std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVD): Tensor " << tensor_name << " not found in tensor contraction: "
                 << contraction << std::endl;
      }
     }else{
      std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVD): Invalid argument#1 in tensor contraction: "
                << contraction << std::endl;
     }
    }else{
     parsed = false;
     std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVD): Tensor " << tensor_name << " not found in tensor contraction: "
               << contraction << std::endl;
    }
   }else{
    std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVD): Invalid argument#0 in tensor contraction: "
              << contraction << std::endl;
   }
  }else{
   parsed = false;
   std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVD): Invalid number of arguments in tensor contraction: "
             << contraction << std::endl;
  }
 }else{
  std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVD): Invalid tensor contraction: " << contraction << std::endl;
 }
 return parsed;
}

bool NumServer::orthogonalizeTensorSVDSync(const std::string & contraction)
{
 std::vector<std::string> tensors;
 auto parsed = parse_tensor_network(contraction,tensors);
 if(parsed){
  if(tensors.size() == 3){
   std::string tensor_name;
   std::vector<IndexLabel> indices;
   bool complex_conj0,complex_conj1,complex_conj2;
   parsed = parse_tensor(tensors[0],tensor_name,indices,complex_conj0);
   if(parsed){
    assert(!complex_conj0);
    auto iter = tensors_.find(tensor_name);
    if(iter != tensors_.end()){
     auto tensor0 = iter->second;
     parsed = parse_tensor(tensors[1],tensor_name,indices,complex_conj1);
     if(parsed){
      assert(!complex_conj1);
      iter = tensors_.find(tensor_name);
      if(iter != tensors_.end()){
       auto tensor1 = iter->second;
       parsed = parse_tensor(tensors[2],tensor_name,indices,complex_conj2);
       if(parsed){
        assert(!complex_conj2);
        iter = tensors_.find(tensor_name);
        if(iter != tensors_.end()){
         auto tensor2 = iter->second;
         std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::ORTHOGONALIZE_SVD);
         op->setTensorOperand(tensor0,complex_conj0);
         op->setIndexPattern(contraction);
         parsed = sync(*tensor0);
         if(parsed) parsed = submit(op);
         if(parsed) parsed = sync(*op);
        }else{
         parsed = false;
         std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVDSync): Tensor " << tensor_name << " not found in tensor contraction: "
                   << contraction << std::endl;
        }
       }else{
        std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVDSync): Invalid argument#2 in tensor contraction: "
                  << contraction << std::endl;
       }
      }else{
       parsed = false;
       std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVDSync): Tensor " << tensor_name << " not found in tensor contraction: "
                 << contraction << std::endl;
      }
     }else{
      std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVDSync): Invalid argument#1 in tensor contraction: "
                << contraction << std::endl;
     }
    }else{
     parsed = false;
     std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVDSync): Tensor " << tensor_name << " not found in tensor contraction: "
               << contraction << std::endl;
    }
   }else{
    std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVDSync): Invalid argument#0 in tensor contraction: "
              << contraction << std::endl;
   }
  }else{
   parsed = false;
   std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVDSync): Invalid number of arguments in tensor contraction: "
             << contraction << std::endl;
  }
 }else{
  std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorSVDSync): Invalid tensor contraction: " << contraction << std::endl;
 }
 return parsed;
}

bool NumServer::orthogonalizeTensorMGS(const std::string & name)
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorMGS): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::ORTHOGONALIZE_MGS);
 op->setTensorOperand(iter->second);
 //`Finish: Convert the isometry specification into a symbolic index pattern
 //op->setIndexPattern(contraction);
 bool parsed = submit(op);
 return parsed;
}

bool NumServer::orthogonalizeTensorMGSSync(const std::string & name)
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()){
  std::cout << "#ERROR(exatn::NumServer::orthogonalizeTensorMGSSync): Tensor " << name << " not found!" << std::endl;
  return false;
 }
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::ORTHOGONALIZE_MGS);
 op->setTensorOperand(iter->second);
 //`Finish: Convert the isometry specification into a symbolic index pattern
 //op->setIndexPattern(contraction);
 bool parsed = submit(op);
 if(parsed) parsed = sync(*op);
 return parsed;
}

bool NumServer::printTensor(const std::string & name){
 return transformTensor(name,std::shared_ptr<TensorMethod>(new numerics::FunctorPrint()));
}

bool NumServer::printTensorSync(const std::string & name){
 return transformTensorSync(name,std::shared_ptr<TensorMethod>(new numerics::FunctorPrint()));
}

bool NumServer::printTensorFile(const std::string & name, const std::string & filename){
 return transformTensor(name,std::shared_ptr<TensorMethod>(new numerics::FunctorPrint(filename)));
}

bool NumServer::printTensorFileSync(const std::string & name, const std::string & filename){
 return transformTensorSync(name,std::shared_ptr<TensorMethod>(new numerics::FunctorPrint(filename)));
}

bool NumServer::evaluateTensorNetwork(const std::string & name,
                                      const std::string & network)
{
 return evaluateTensorNetwork(getDefaultProcessGroup(),name,network);
}

bool NumServer::evaluateTensorNetwork(const ProcessGroup & process_group,
                                      const std::string & name,
                                      const std::string & network)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
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
  }
  if(parsed){
   TensorNetwork tensnet(name,network,tensor_map);
   parsed = submit(process_group,tensnet);
  }
 }else{
  std::cout << "#ERROR(exatn::NumServer::evaluateTensorNetwork): Invalid tensor network: " << network << std::endl;
 }
 return parsed;
}

bool NumServer::evaluateTensorNetworkSync(const std::string & name,
                                          const std::string & network)
{
 return evaluateTensorNetworkSync(getDefaultProcessGroup(),name,network);
}

bool NumServer::evaluateTensorNetworkSync(const ProcessGroup & process_group,
                                          const std::string & name,
                                          const std::string & network)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
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
  }
  if(parsed){
   TensorNetwork tensnet(name,network,tensor_map);
   parsed = submit(process_group,tensnet);
   if(parsed) parsed = sync(process_group,tensnet);
  }
 }else{
  std::cout << "#ERROR(exatn::NumServer::evaluateTensorNetworkSync): Invalid tensor network: " << network << std::endl;
 }
 return parsed;
}

bool NumServer::normalizeNorm2Sync(const std::string & name, double norm)
{
 double old_norm = 0.0;
 bool success = computeNorm2Sync(name,old_norm);
 if(success) success = scaleTensorSync(name,norm/old_norm);
 return success;
}

bool NumServer::normalizeNorm2Sync(TensorExpansion & expansion,
                                   double norm)
{
 return normalizeNorm2Sync(getDefaultProcessGroup(),expansion,norm);
}

bool NumServer::normalizeNorm2Sync(const ProcessGroup & process_group,
                                   TensorExpansion & expansion,
                                   double norm)
{
 //Determine parallel execution configuration:
 unsigned int local_rank; //local process rank within the process group
 if(!process_group.rankIsIn(process_rank_,&local_rank)) return true; //process is not in the group: Do nothing
 //Perform normalization:
 bool success = false;
 if(expansion.getNumComponents() > 0){
  auto inner_prod_tensor = exatn::makeSharedTensor("_InnerProd");
  assert(expansion.cbegin()->network->getTensorElementType() != TensorElementType::VOID);
  success = createTensor(inner_prod_tensor,expansion.cbegin()->network->getTensorElementType());
  if(success){
   success = initTensor("_InnerProd",0.0);
   if(success){
    TensorExpansion conjugate(expansion);
    conjugate.conjugate();
    conjugate.rename(expansion.getName()+"Conj");
    TensorExpansion inner_product(conjugate,expansion);
    inner_product.rename("InnerProduct");
    success = sync("_InnerProd"); assert(success);
    success = submit(process_group,inner_product,inner_prod_tensor);
    if(success){
     success = sync("_InnerProd"); assert(success);
     double original_norm2;
     success = computeNorm1Sync("_InnerProd",original_norm2);
     if(success){
      if(original_norm2 > 0.0){
       expansion.rescale(std::complex<double>(norm/std::sqrt(original_norm2)));
      }else{
       std::cout << "#WARNING(exatn::normalizeNorm2): Tensor expansion has zero norm, thus cannot be renormalized!" << std::endl;
       success = false;
      }
     }else{
      std::cout << "#ERROR(exatn::normalizeNorm2): Unable to compute the norm!" << std::endl;
     }
    }else{
     std::cout << "#ERROR(exatn::normalizeNorm2): Unable to evaluate the inner product tensor expansion!" << std::endl;
    }
   }else{
    std::cout << "#ERROR(exatn::normalizeNorm2): Unable to zero out the scalar tensor!" << std::endl;
   }
   bool done = destroyTensorSync("_InnerProd"); success = (success && done);
   if(!done) std::cout << "#ERROR(exatn::normalizeNorm2): Unable to destroy the scalar tensor!" << std::endl;
  }else{
   std::cout << "#ERROR(exatn::normalizeNorm2): Unable to create the scalar tensor!" << std::endl;
  }
 }
 return success;
}

bool NumServer::balanceNorm2Sync(TensorNetwork & network,
                                 double norm,
                                 bool only_optimizable)
{
 return balanceNorm2Sync(getDefaultProcessGroup(),network,norm,only_optimizable);
}

bool NumServer::balanceNorm2Sync(const ProcessGroup & process_group,
                                 TensorNetwork & network,
                                 double norm,
                                 bool only_optimizable)
{
 unsigned int local_rank; //local process rank within the process group
 if(!process_group.rankIsIn(process_rank_,&local_rank)) return true; //process is not in the group: Do nothing
 bool success = true;
 for(auto tens = network.begin(); tens != network.end(); ++tens){
  if(tens->first != 0){ //output tensor is ignored
   if(!only_optimizable || tens->second.isOptimizable()){ //only optimizable tensors may be renormalized
    double tens_norm;
    success = computeNorm2Sync(tens->second.getName(),tens_norm);
    if(success){
     if(tens_norm > 0.0){
      success = scaleTensorSync(tens->second.getName(),norm/tens_norm);
      if(!success){
       std::cout << "#ERROR(exatn::balanceNorm2): Unable to rescale input tensor "
                 << tens->second.getName() << std::endl;
       break;
      }
     }else{
      std::cout << "#WARNING(exatn::balanceNorm2): Tensor has zero norm, thus cannot be renormalized!" << std::endl;
     }
    }else{
     std::cout << "#ERROR(exatn::balanceNorm2): Unable to compute the norm of input tensor "
               << tens->second.getName() << std::endl;
     break;
    }
   }
  }
 }
 return success;
}

bool NumServer::balanceNorm2Sync(TensorExpansion & expansion,
                                 double norm,
                                 bool only_optimizable)
{
 return balanceNorm2Sync(getDefaultProcessGroup(),expansion,norm,only_optimizable);
}

bool NumServer::balanceNorm2Sync(const ProcessGroup & process_group,
                                 TensorExpansion & expansion,
                                 double norm,
                                 bool only_optimizable)
{
 unsigned int local_rank; //local process rank within the process group
 if(!process_group.rankIsIn(process_rank_,&local_rank)) return true; //process is not in the group: Do nothing
 bool success = true;
 for(auto net = expansion.begin(); net != expansion.end(); ++net){
  success = balanceNorm2Sync(process_group,*(net->network),norm,only_optimizable);
  if(!success) break;
 }
 return success;
}

bool NumServer::balanceNormalizeNorm2Sync(TensorExpansion & expansion,
                                          double tensor_norm,
                                          double expansion_norm,
                                          bool only_optimizable)
{
 return balanceNormalizeNorm2Sync(getDefaultProcessGroup(),expansion,tensor_norm,expansion_norm,only_optimizable);
}

bool NumServer::balanceNormalizeNorm2Sync(const ProcessGroup & process_group,
                                          TensorExpansion & expansion,
                                          double tensor_norm,
                                          double expansion_norm,
                                          bool only_optimizable)
{
 bool success = balanceNorm2Sync(process_group,expansion,tensor_norm,only_optimizable);
 if(success) success = normalizeNorm2Sync(process_group,expansion,expansion_norm);
 return success;
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
  int ref_count = 1;
  auto tens = tensors_.find((*iter)->getName());
  if(tens != tensors_.end()) ++ref_count;
  if(iter->use_count() <= ref_count){
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

void NumServer::printImplicitTensors() const
{
 std::cout << "#DEBUG(exatn::NumServer::printImplicitTensors):" << std::endl;
 for(const auto & tens: implicit_tensors_){
  std::cout << tens->getName() << ": Reference count = " << tens.use_count() << std::endl;
 }
 std::cout << "#END" << std::endl << std::flush;
 return;
}

} //namespace exatn
