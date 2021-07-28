/** ExaTN::Numerics: Tensor operation: Adds a tensor to another tensor
REVISION: 2021/07/28

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_add.hpp"

#include "tensor_node_executor.hpp"

#include <limits>
#include <stack>
#include <algorithm>

namespace exatn{

namespace numerics{

TensorOpAdd::TensorOpAdd():
 TensorOperation(TensorOpCode::ADD,2,1,1+0*2,{0,1})
{
 this->setScalar(0,std::complex<double>{1.0,0.0}); //default alpha prefactor
}

bool TensorOpAdd::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands() && this->getIndexPattern().length() > 0);
}

int TensorOpAdd::accept(runtime::TensorNodeExecutor & node_executor,
                        runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

double TensorOpAdd::getFlopEstimate() const
{
 if(this->isSet()) return static_cast<double>(this->getTensorOperand(0)->getVolume()); //FMA flops (without FMA factor)
 return 0.0;
}

std::unique_ptr<TensorOperation> TensorOpAdd::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpAdd());
}

std::size_t TensorOpAdd::decompose(const TensorMapper & tensor_mapper)
{
 const bool optimized_communication = true; //activates tensor operation sorting for optimized communication

 if(this->isComposite()){
  if(simple_operations_.empty()){
   const auto num_procs = tensor_mapper.getNumProcesses();
   const auto proc_rank = tensor_mapper.getProcessRank();
   const auto & intra_comm = tensor_mapper.getMPICommProxy();
   std::vector<std::shared_ptr<TensorOperation>> simple_operations;
   std::vector<std::pair<int, std::size_t>> comm_distance; //{communication distance, operation position}

   auto process_distance = [&num_procs](unsigned int proc0, unsigned int proc1){
    int dist = proc0 <= proc1 ? static_cast<int>(proc1-proc0) :
                                static_cast<int>(num_procs-(proc0-proc1));
    return dist;
   };

   auto comp_lt = [](const std::pair<int, std::size_t> & item0,
                     const std::pair<int, std::size_t> & item1){
    return (item0.first < item1.first);
   };

   //Prepare composite tensor operands:
   auto tensor0 = getTensorOperand(0);
   auto tensor1 = getTensorOperand(1);
   auto comp_tens0 = castTensorComposite(tensor0);
   auto comp_tens1 = castTensorComposite(tensor1);
   assert(comp_tens0 || comp_tens1);
   const auto tens_elem_type0 = tensor0->getElementType();
   const auto tens_elem_type1 = tensor1->getElementType();
   std::map<unsigned long long, std::shared_ptr<Tensor>> self_composite0 {{0,tensor0}};
   std::map<unsigned long long, std::shared_ptr<Tensor>> self_composite1 {{0,tensor1}};
   auto beg0 = self_composite0.begin();
   auto end0 = self_composite0.end();
   unsigned long long num_subtensors0 = 1;
   if(comp_tens0){
    beg0 = comp_tens0->begin();
    end0 = comp_tens0->end();
    num_subtensors0 = comp_tens0->getNumSubtensors();
   }
   auto beg1 = self_composite1.begin();
   auto end1 = self_composite1.end();
   unsigned long long num_subtensors1 = 1;
   if(comp_tens1){
    beg1 = comp_tens1->begin();
    end1 = comp_tens1->end();
    num_subtensors1 = comp_tens1->getNumSubtensors();
   }
   //Send local tensor slices to remote processes:
   std::stack<std::shared_ptr<Tensor>> slices; //temporary tensor slices
   for(auto subtens1 = beg1; subtens1 != end1; ++subtens1){
    if(tensor_mapper.isLocalSubtensor(*(subtens1->second))){
     for(auto subtens0 = beg0; subtens0 != end0; ++subtens0){
      std::shared_ptr<Tensor> slice;
      bool congruent = subtens0->second->isCongruentTo(*(subtens1->second));
      if(congruent){
       slice = subtens1->second;
      }else{
       slice = makeSharedTensorIntersection("_",*(subtens0->second),*(subtens1->second));
       if(slice) slice->rename();
      }
      if(slice){
       if(!congruent){
        {//Create the tensor intersection slice:
         comm_distance.emplace_back(std::make_pair(-1,simple_operations.size()));
         simple_operations.emplace_back(std::move(TensorOpCreate::createNew()));
         auto & op = simple_operations.back();
         op->setTensorOperand(slice);
         std::dynamic_pointer_cast<TensorOpCreate>(op)->resetTensorElementType(tens_elem_type1);
        }
        slices.emplace(slice); //stack of slices to be destroyed later
        {//Extract the intersection slice:
         comm_distance.emplace_back(std::make_pair(-1,simple_operations.size()));
         simple_operations.emplace_back(std::move(TensorOpSlice::createNew()));
         auto & op = simple_operations.back();
         op->setTensorOperand(slice);
         op->setTensorOperand(subtens1->second);
        }
       }
       const auto tensor_owner_first = tensor_mapper.subtensorFirstOwnerId(subtens0->first,num_subtensors0);
       for(auto tensor_owner = tensor_owner_first; tensor_owner < num_procs; tensor_owner += num_subtensors0){
        if(tensor_owner != proc_rank){ //consider only remote processes
         if(tensor_mapper.subtensorOwnerId(tensor_owner,subtens1->first,num_subtensors1) == proc_rank){//Upload the intersection slice to a remote process:
          comm_distance.emplace_back(std::make_pair(process_distance(proc_rank,tensor_owner),simple_operations.size()));
          simple_operations.emplace_back(std::move(TensorOpUpload::createNew()));
          auto & op = simple_operations.back();
          op->setTensorOperand(slice);
          std::dynamic_pointer_cast<TensorOpUpload>(op)->resetMPICommunicator(intra_comm);
          auto success = std::dynamic_pointer_cast<TensorOpUpload>(op)->resetRemoteProcessRank(tensor_owner); assert(success);
          assert(subtens0->first <= std::numeric_limits<int>::max());
          success = std::dynamic_pointer_cast<TensorOpUpload>(op)->resetMessageTag(static_cast<int>(subtens0->first)); assert(success);
         }
        }
       }
      }
     }
    }
   }
   //Receive necessary tensor slices from remote processes:
   for(auto subtens0 = beg0; subtens0 != end0; ++subtens0){
    if(tensor_mapper.isLocalSubtensor(*(subtens0->second))){
     for(auto subtens1 = beg1; subtens1 != end1; ++subtens1){
      std::shared_ptr<Tensor> slice;
      bool congruent = subtens0->second->isCongruentTo(*(subtens1->second)) &&
                       tensor_mapper.isLocalSubtensor(*(subtens1->second));
      if(congruent){
       slice = subtens1->second;
      }else{
       slice = makeSharedTensorIntersection("_",*(subtens0->second),*(subtens1->second));
       if(slice) slice->rename();
      }
      if(slice){
       const auto tensor_owner = tensor_mapper.subtensorOwnerId(subtens1->first,num_subtensors1);
       if(!congruent){//Create the tensor intersection slice:
        comm_distance.emplace_back(std::make_pair(-1,simple_operations.size()));
        simple_operations.emplace_back(std::move(TensorOpCreate::createNew()));
        auto & op = simple_operations.back();
        op->setTensorOperand(slice);
        std::dynamic_pointer_cast<TensorOpCreate>(op)->resetTensorElementType(tens_elem_type0);
        slices.emplace(slice); //stack of slices to be destroyed later
       }
       if(tensor_owner == tensor_mapper.getProcessRank()){ //slice is local
        if(!congruent){//Extract the (local) intersection slice:
         comm_distance.emplace_back(std::make_pair(-1,simple_operations.size()));
         simple_operations.emplace_back(std::move(TensorOpSlice::createNew()));
         auto & op = simple_operations.back();
         op->setTensorOperand(slice);
         op->setTensorOperand(subtens1->second);
        }
       }else{ //slice is remote
        //Fetch the (remote) intersection slice:
        comm_distance.emplace_back(std::make_pair(process_distance(tensor_owner,proc_rank),simple_operations.size()));
        simple_operations.emplace_back(std::move(TensorOpFetch::createNew()));
        auto & op = simple_operations.back();
        op->setTensorOperand(slice);
        std::dynamic_pointer_cast<TensorOpFetch>(op)->resetMPICommunicator(intra_comm);
        auto success = std::dynamic_pointer_cast<TensorOpFetch>(op)->resetRemoteProcessRank(tensor_owner); assert(success);
        assert(subtens0->first <= std::numeric_limits<int>::max());
        success = std::dynamic_pointer_cast<TensorOpFetch>(op)->resetMessageTag(static_cast<int>(subtens0->first)); assert(success);
       }
       {//Insert the intersection slice:
        comm_distance.emplace_back(std::make_pair(static_cast<int>(num_procs+1),simple_operations.size()));
        simple_operations.emplace_back(std::move(TensorOpInsert::createNew()));
        auto & op = simple_operations.back();
        op->setTensorOperand(subtens0->second);
        op->setTensorOperand(slice);
        std::dynamic_pointer_cast<TensorOpInsert>(op)->resetAccumulative(true);
       }
      }
     }
    }
   }
   //Destroy temporary slices:
   while(!slices.empty()){
    comm_distance.emplace_back(std::make_pair(static_cast<int>(num_procs+2),simple_operations.size()));
    simple_operations.emplace_back(std::move(TensorOpDestroy::createNew()));
    auto & op = simple_operations.back();
    op->setTensorOperand(slices.top());
    slices.pop();
   }
   //Sort generated simple tensor operations for optimal execution:
   if(optimized_communication){
    const auto num_ops = simple_operations.size();
    assert(comm_distance.size() == num_ops);
    std::stable_sort(comm_distance.begin(),comm_distance.end(),comp_lt);
    simple_operations_.resize(num_ops);
    for(std::size_t i = 0; i < num_ops; ++i) simple_operations_[i] = std::move(simple_operations[comm_distance[i].second]);
   }else{
    simple_operations_ = std::move(simple_operations);
   }
  }
 }
 return simple_operations_.size();
}

} //namespace numerics

} //namespace exatn
