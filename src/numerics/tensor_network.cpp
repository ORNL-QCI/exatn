/** ExaTN::Numerics: Tensor network
REVISION: 2022/07/14

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corporation

SPDX-License-Identifier: BSD-3-Clause **/

#include "tensor_network.hpp"
#include "tensor_symbol.hpp"
#include "functor_init_val.hpp"
#include "contraction_seq_optimizer_factory.hpp"
#include "metis_graph.hpp"

#include <iostream>
#include <unordered_set>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <memory>
#include <algorithm>

namespace exatn{

namespace numerics{

//Tensor contraction sequence optmizers:
std::map<std::string,std::shared_ptr<ContractionSeqOptimizer>> optimizers;


//Helpers:
inline bool isIntermediateTensorName(const std::string & tensor_name)
{
 if(tensor_name.length() >= 2){
  if((tensor_name[0] == '_' && tensor_name[1] == 'x') ||
     (tensor_name[0] == '_' && tensor_name[1] == 'y') ||
     (tensor_name[0] == '_' && tensor_name[1] == 'z')) return true;
 }
 return false;
}


inline bool isPureIntermediateTensorName(const std::string & tensor_name)
{
 if(tensor_name.length() >= 2){
  if((tensor_name[0] == '_' && tensor_name[1] == 'x') ||
     (tensor_name[0] == '_' && tensor_name[1] == 'y')) return true;
 }
 return false;
}


bool tensorNameIsIntermediate(const Tensor & tensor,
                              bool * network_output)
{
 bool res = false, out = false;
 const auto & tens_name = tensor.getName();
 if(tens_name.length() >= 2){
  out = (tens_name[0] == '_' && tens_name[1] == 'z');    //_z: output tensor of the tensor network
  res = (out ||                                          //output tensor is also considered intermediate
         (tens_name[0] == '_' && tens_name[1] == 'y') || //_y: intermediate tensor of the tensor network
         (tens_name[0] == '_' && tens_name[1] == 'x'));  //_x: intermediate tensor of the tensor network
 }
 if(network_output != nullptr) *network_output = out;
 return res;
}


double getTensorContractionCost(const TensorConn & left_tensor, const TensorConn & right_tensor,
                                double * total_volume, double * diff_volume,
                                double * arithm_intensity, bool adjust_cost)
{
 double flops = 0.0, left_vol = 1.0, right_vol = 1.0, contr_vol = 1.0;
 const auto left_id = left_tensor.getTensorId();
 const auto left_rank = left_tensor.getNumLegs();
 const auto right_id = right_tensor.getTensorId();
 const auto right_rank = right_tensor.getNumLegs();
 const auto & right_legs = right_tensor.getTensorLegs();
 for(unsigned int i = 0; i < left_rank; ++i){
  left_vol *= static_cast<double>(left_tensor.getDimExtent(i));
 }
 for(unsigned int i = 0; i < right_rank; ++i){
  double dim_ext = static_cast<double>(right_tensor.getDimExtent(i));
  if(right_legs[i].getTensorId() == left_id) contr_vol *= dim_ext; //contracted dimension
  right_vol *= dim_ext;
 }
 flops = left_vol * right_vol / contr_vol; //FMA flops (no FMA prefactor)
 double tot_vol = left_vol + right_vol + (flops / contr_vol); //total volume of tensors
 if(total_volume != nullptr) *total_volume = tot_vol;
 if(diff_volume != nullptr) *diff_volume = (flops / contr_vol) - (left_vol + right_vol);
 if(arithm_intensity != nullptr) *arithm_intensity = flops / tot_vol;
 if(adjust_cost){ //increase the "effective" flop count if arithmetic intensity is low
  //`Finish: flops *= f(arithm_intensity): [max --> 1]
 }
 return flops;
}


void printContractionSequence(const std::list<numerics::ContrTriple> & contr_seq)
{
 unsigned int i = 0;
 for(const auto & contr: contr_seq){
  std::cout << "{" << contr.result_id << ":" << contr.left_id << "," << contr.right_id << "}";
  if(++i == 10){std::cout << std::endl; i = 0;}
 }
 if(i != 0) std::cout << std::endl;
 return;
}


void printContractionSequence(std::ofstream & output_file, const std::list<numerics::ContrTriple> & contr_seq)
{
 unsigned int i = 0;
 for(const auto & contr: contr_seq){
  output_file << "{" << contr.result_id << ":" << contr.left_id << "," << contr.right_id << "}";
  if(++i == 10){output_file << std::endl; i = 0;}
 }
 if(i != 0) output_file << std::endl;
 return;
}


//Main:
TensorNetwork::TensorNetwork():
 explicit_output_(0), finalized_(1), has_isometries_(0), max_tensor_id_(0),
 contraction_seq_flops_(0.0), max_intermediate_presence_volume_(0.0),
 max_intermediate_volume_(0.0), max_intermediate_rank_(0), universal_indexing_(false)
{
 auto res = emplaceTensorConnDirect(false,
                                    0U, //output tensor (id = 0)
                                    std::make_shared<Tensor>("_smoky"),0U,std::vector<TensorLeg>{});
 if(!res){
  std::cout << "#ERROR(exatn::numerics::TensorNetwork::TensorNetwork): Tensor id already in use!" << std::endl;
  assert(false);
 }
}


TensorNetwork::TensorNetwork(const std::string & name):
 explicit_output_(0), finalized_(1), name_(name), has_isometries_(0), max_tensor_id_(0),
 contraction_seq_flops_(0.0), max_intermediate_presence_volume_(0.0),
 max_intermediate_volume_(0.0), max_intermediate_rank_(0), universal_indexing_(false)
{
 auto res = emplaceTensorConnDirect(false,
                                    0U, //output tensor (id = 0)
                                    std::make_shared<Tensor>(name),0U,std::vector<TensorLeg>{});
 if(!res){
  std::cout << "#ERROR(exatn::numerics::TensorNetwork::TensorNetwork): Tensor id already in use!" << std::endl;
  assert(false);
 }
}


TensorNetwork::TensorNetwork(const std::string & name,
                             std::shared_ptr<Tensor> output_tensor,
                             const std::vector<TensorLeg> & output_legs):
 explicit_output_(1), finalized_(0), name_(name), has_isometries_(0), max_tensor_id_(0),
 contraction_seq_flops_(0.0), max_intermediate_presence_volume_(0.0),
 max_intermediate_volume_(0.0), max_intermediate_rank_(0), universal_indexing_(false)
{
 auto res = emplaceTensorConnDirect(false,
                                    0U, //output tensor (id = 0)
                                    output_tensor,0U,output_legs);
 if(!res){
  std::cout << "#ERROR(exatn::numerics::TensorNetwork::TensorNetwork): Tensor id already in use!" << std::endl;
  assert(false);
 }
}


TensorNetwork::TensorNetwork(const std::string & name,
                             const std::string & tensor_network,
                             const std::map<std::string,std::shared_ptr<Tensor>> & tensors):
 explicit_output_(1), finalized_(0), name_(name), has_isometries_(0), max_tensor_id_(0),
 contraction_seq_flops_(0.0), max_intermediate_presence_volume_(0.0),
 max_intermediate_volume_(0.0), max_intermediate_rank_(0), universal_indexing_(false)
{
 //Convert tensor hypernetwork into regular tensor network, if needed:
 //`Finish
 //Build a regular tensor network according the the provided symbolic specification:
 std::map<std::string,std::vector<TensorLeg>> index_map; //index label --> list of tensor legs associated with this index label
 std::vector<std::string> stensors; //individual tensors of the tensor network (symbolic)
 if(parse_tensor_network(tensor_network,stensors)){
  //Construct index correspondence map:
  std::string tensor_name;
  std::vector<IndexLabel> indices;
  for(unsigned int i = 0; i < stensors.size(); ++i){
   bool conjugated;
   if(parse_tensor(stensors[i],tensor_name,indices,conjugated)){
    for(unsigned int j = 0; j < indices.size(); ++j){
     auto pos = index_map.find(indices[j].label);
     if(pos == index_map.end()){
      auto res = index_map.emplace(std::make_pair(indices[j].label,std::vector<TensorLeg>{}));
      assert(res.second);
      pos = res.first;
     }
     pos->second.emplace_back(TensorLeg(i,j,indices[j].direction)); //directed index #j of tensor #i
    }
   }else{
    std::cout << "#ERROR(TensorNetwork::TensorNetwork): Invalid tensor in symbolic tensor network specification: " <<
     stensors[i] << std::endl;
    assert(false);
   }
   indices.clear();
   tensor_name.clear();
  }
  //Build the tensor network object:
  for(unsigned int i = 0; i < stensors.size(); ++i){
   bool conjugated;
   if(parse_tensor(stensors[i],tensor_name,indices,conjugated)){
    auto tensor = tensors.find(tensor_name);
    if(tensor != tensors.end()){
     std::vector<TensorLeg> legs;
     for(unsigned int j = 0; j < indices.size(); ++j){
      auto pos = index_map.find(indices[j].label);
      assert(pos != index_map.end());
      const auto & inds = pos->second;
      for(const auto & ind: inds){
       if(ind.getTensorId() != i || ind.getDimensionId() != j){
        legs.emplace_back(ind);
       }
      }
     }
     if(i == 0){
      assert(!conjugated); //output tensor must not appear complex conjugated
      auto res = emplaceTensorConnDirect(false,
                                         0U, //output tensor (id = 0)
                                         tensor->second,0U,legs);
      if(!res){
       std::cout << "#ERROR(exatn::numerics::TensorNetwork::TensorNetwork): Tensor id already in use!" << std::endl;
       assert(false);
      }
     }else{ //input tensor
      this->placeTensor(i,tensor->second,legs,conjugated);
     }
    }else{
     std::cout << "#ERROR(TensorNetwork::TensorNetwork): Unable to find tensor named " <<
      tensor_name << std::endl;
     assert(false);
    }
   }
  }
 }else{
  std::cout << "#ERROR(TensorNetwork::TensorNetwork): Invalid symbolic tensor network specification: " <<
   tensor_network << std::endl;
  assert(false);
 }
 bool finalized = this->finalize();
 assert(finalized);
}


TensorNetwork::TensorNetwork(const std::string & name,
                             std::shared_ptr<Tensor> output_tensor,
                             NetworkBuilder & builder,
                             bool tensor_operator):
 explicit_output_(1), finalized_(0), name_(name), has_isometries_(0), max_tensor_id_(0),
 contraction_seq_flops_(0.0), max_intermediate_presence_volume_(0.0),
 max_intermediate_volume_(0.0), max_intermediate_rank_(0), universal_indexing_(false)
{
 auto new_out_tensor = output_tensor->clone();
 new_out_tensor->rename(tensor_hex_name("z",new_out_tensor->getTensorHash()));
 auto res = emplaceTensorConnDirect(false,
                                    0U, //output tensor (id = 0)
                                    new_out_tensor,0U,
                                    std::vector<TensorLeg>(output_tensor->getRank(),TensorLeg(0,0))); //dummy legs
 if(!res){
  std::cout << "#ERROR(exatn::numerics::TensorNetwork::TensorNetwork): Tensor id already in use!" << std::endl;
  assert(false);
 }
 builder.build(*this,tensor_operator); //create and link input tensors of the tensor network
 finalized_ = 1;
 updateConnectionsFromInputTensors(); //update output tensor legs
}


TensorNetwork::TensorNetwork(const TensorNetwork & another,
                             bool replace_output,
                             const std::string & new_output_name)
{
 *this = another;
 if(replace_output) this->resetOutputTensor(new_output_name);
}


TensorNetwork::TensorNetwork(const std::string & name,
                             const TensorNetwork & another,
                             const std::vector<unsigned int> & tensor_ids):
 TensorNetwork(name)
{
 //Check tensor ids:
 std::unordered_set<unsigned int> ids;
 for(const auto tens_id: tensor_ids){
  assert(tens_id != 0);
  auto res = ids.emplace(tens_id);
  assert(res.second);
 }
 //Copy the original output tensor:
 auto success = emplaceTensorConn(0,*(const_cast<TensorNetwork&>(another).getTensorConn(0)));
 assert(success);
 auto * out_tens_conn = getTensorConn(0);
 out_tens_conn->replaceStoredTensor();
 //Append input tensors of the sub-network:
 for(const auto tens_id: tensor_ids){
  auto * tens_conn = const_cast<TensorNetwork&>(another).getTensorConn(tens_id);
  assert(tens_conn != nullptr);
  success = emplaceTensorConn(tens_id,*tens_conn); assert(success);
 }
 //Delete output tensor legs not associated with the chosen sub-network:
 int leg_id = 0;
 while(leg_id < out_tens_conn->getNumLegs()){
  auto leg = out_tens_conn->getTensorLeg(leg_id);
  auto it = ids.find(leg.getTensorId());
  if(it == ids.cend()){
   out_tens_conn->deleteLeg(leg_id);
  }else{
   ++leg_id;
  }
 }
 finalized_ = 1;
 updateConnections(0);
 //Append boundary legs of the sub-network to the output tensor:
 unsigned int n = out_tens_conn->getNumLegs();
 for(const auto tens_id: tensor_ids){
  auto * tens_conn = getTensorConn(tens_id);
  const unsigned int num_legs = tens_conn->getNumLegs();
  for(unsigned int i = 0; i < num_legs; ++i){
   auto leg = tens_conn->getTensorLeg(i);
   auto other_tensor_id = leg.getTensorId();
   if(other_tensor_id != 0){
    auto it = ids.find(other_tensor_id);
    if(it == ids.cend()){ //boundary leg
     leg.resetTensorId(0);
     leg.resetDimensionId(n++);
     tens_conn->resetLeg(i,leg);
     leg.resetTensorId(tens_id);
     leg.resetDimensionId(i);
     leg.reverseDirection();
     out_tens_conn->appendLeg(tens_conn->getDimSpaceAttr(i),
                              tens_conn->getDimExtent(i),
                              leg);
    }
   }
  }
 }
}


void TensorNetwork::printIt(bool with_tensor_hash) const
{
 std::cout << "TensorNetwork(" << name_
           << ")[rank = " << this->getRank()
           << ", size = " << this->getNumTensors() << "]{" << std::endl;
 for(const auto & kv: tensors_){
  std::cout << " ";
  kv.second.printIt(with_tensor_hash);
 }
 std::cout << "}" << std::endl;
 return;
}


void TensorNetwork::printItFile(std::ofstream & output_file,
                                bool with_tensor_hash) const
{
 output_file << "TensorNetwork(" << name_
             << ")[rank = " << this->getRank()
             << ", size = " << this->getNumTensors() << "]{" << std::endl;
 for(const auto & kv: tensors_){
  output_file << " ";
  kv.second.printItFile(output_file,with_tensor_hash);
 }
 output_file << "}" << std::endl;
 return;
}


bool TensorNetwork::isEmpty() const
{
 return (tensors_.size() <= 1); //only output tensor exists => still empty
}


bool TensorNetwork::isExplicit() const
{
 return (explicit_output_ != 0);
}


bool TensorNetwork::isFinalized() const
{
 return (finalized_ != 0);
}


bool TensorNetwork::isValid()
{
 return checkConnections();
}


unsigned int TensorNetwork::getRank() const
{
 //assert(this->isFinalized());
 return tensors_.at(0).getNumLegs(); //output tensor
}


unsigned int TensorNetwork::getNumTensors() const
{
 return static_cast<unsigned int>(tensors_.size() - 1); //output tensor is not counted
}


unsigned int TensorNetwork::getMaxTensorId()
{
 if(max_tensor_id_ == 0){
  for(const auto & kv: tensors_) max_tensor_id_ = std::max(max_tensor_id_,kv.first);
 }
 return max_tensor_id_;
}


TensorElementType TensorNetwork::getTensorElementType() const
{
 assert(this->isFinalized());
 for(const auto & tens: tensors_){
  if(tens.first != 0){
   const auto elem_type = tens.second.getElementType();
   if(elem_type != TensorElementType::VOID) return elem_type;
  }
 }
 return TensorElementType::VOID;
}


void TensorNetwork::updateMaxTensorIdOnAppend(unsigned int tensor_id)
{
 auto curr_max_id = getMaxTensorId();
 max_tensor_id_ = std::max(curr_max_id,tensor_id);
 return;
}


void TensorNetwork::updateMaxTensorIdOnRemove(unsigned int tensor_id)
{
 if(tensor_id != 0 && tensor_id == max_tensor_id_){
  max_tensor_id_ = 0; //reset max tensor id to Undefined
  //auto refresh_max_tensor_id = getMaxTensorId();
 }
 return;
}


void TensorNetwork::resetOutputTensor(const std::string & name)
{
 assert(finalized_ != 0);
 auto iter = tensors_.find(0);
 assert(iter != tensors_.end());
 iter->second.replaceStoredTensor(name);
 return;
}


void TensorNetwork::resetOutputTensor(const std::vector<unsigned int> & order,
                                      const std::string & name)
{
 assert(finalized_ != 0);
 auto iter = tensors_.find(0);
 assert(iter != tensors_.end());
 iter->second.replaceStoredTensor(order,name);
 return;
}


const std::string & TensorNetwork::getName() const
{
 return name_;
}


void TensorNetwork::rename(const std::string & name)
{
 assert(finalized_ != 0);
 resetOutputTensor();
 name_ = name;
 return;
}


TensorConn * TensorNetwork::getTensorConn(unsigned int tensor_id)
{
 auto it = tensors_.find(tensor_id);
 if(it == tensors_.end()) return nullptr;
 return &(it->second);
}


std::vector<TensorConn*> TensorNetwork::getTensorConnAll()
{
 std::vector<TensorConn*> tensors(this->getNumTensors(),nullptr);
 unsigned int i = 0;
 for(auto & kv: tensors_){
  if(kv.first != 0) tensors[i++] = &(kv.second);
 }
 return tensors;
}


std::shared_ptr<Tensor> TensorNetwork::getTensor(unsigned int tensor_id, bool * conjugated) const
{
 auto it = tensors_.find(tensor_id);
 if(it == tensors_.end()) return std::shared_ptr<Tensor>(nullptr);
 if(conjugated != nullptr) *conjugated = (it->second).isComplexConjugated();
 return (it->second).getTensor();
}


const std::vector<TensorLeg> * TensorNetwork::getTensorConnections(unsigned int tensor_id) const
{
 auto it = tensors_.find(tensor_id);
 if(it == tensors_.end()) return nullptr;
 return &((it->second).getTensorLegs());
}


std::list<unsigned int> TensorNetwork::getAdjacentTensors(unsigned int tensor_id) const
{
 std::list<unsigned int> tensor_ids;
 const auto * legs = this->getTensorConnections(tensor_id);
 if(legs != nullptr){
  for(const auto & leg: *legs){
   const auto new_tensor_id = leg.getTensorId();
   if(new_tensor_id != 0){ //ignore the output tensor
    auto iter = std::find(tensor_ids.begin(),tensor_ids.end(),new_tensor_id);
    if(iter == tensor_ids.end()) tensor_ids.emplace_back(new_tensor_id);
   }
  }
 }
 return tensor_ids;
}


bool TensorNetwork::finalize(bool check_validity)
{
 if(finalized_ == 0){
  if(this->isEmpty()){ //empty networks cannot be finalized
   std::cout << "#ERROR(TensorNetwork::finalize): Empty tensor network cannot be finalized!" << std::endl;
   return false;
  }
  finalized_ = 1;
  if(check_validity){
   if(!checkConnections()){
    finalized_ = 0;
    std::cout << "#ERROR(TensorNetwork::finalize): Invalid connectivity prevents tensor network finalization!" << std::endl;
    return false;
   }
  }
 }
 return true;
}


bool TensorNetwork::checkConnections(unsigned int tensor_id)
{
 assert(finalized_ != 0); //tensor network must be in finalized state
 auto * tensor = this->getTensorConn(tensor_id);
 assert(tensor != nullptr); //invalid tensor_id
 auto tensor_rank = tensor->getNumLegs();
 for(unsigned int i = 0; i < tensor_rank; ++i){
  const auto & tensor_leg = tensor->getTensorLeg(i);
  auto other_tensor_id = tensor_leg.getTensorId();
  auto other_tensor_leg_id = tensor_leg.getDimensionId();
  auto * other_tensor = this->getTensorConn(other_tensor_id);
  assert(other_tensor != nullptr); //unable to find the linked tensor
  const auto & other_tensor_leg = other_tensor->getTensorLeg(other_tensor_leg_id);
  if(other_tensor_leg.getTensorId() != tensor_id ||
     other_tensor_leg.getDimensionId() != i ||
     other_tensor_leg.getDirection() != reverseLegDirection(tensor_leg.getDirection())) return false;
 }
 return true;
}


bool TensorNetwork::checkConnections()
{
 assert(finalized_ != 0); //tensor network must be in finalized state
 for(const auto & kv: tensors_){
  if(!checkConnections(kv.first)) return false;
 }
 return true;
}


void TensorNetwork::updateConnections(unsigned int tensor_id)
{
 assert(finalized_ != 0); //tensor network must be in finalized state
 auto * tensor = this->getTensorConn(tensor_id);
 assert(tensor != nullptr); //invalid tensor_id
 auto tensor_rank = tensor->getNumLegs();
 for(unsigned int i = 0; i < tensor_rank; ++i){
  const auto & tensor_leg = tensor->getTensorLeg(i);
  auto other_tensor_id = tensor_leg.getTensorId();
  auto other_tensor_leg_id = tensor_leg.getDimensionId();
  auto * other_tensor = this->getTensorConn(other_tensor_id);
  assert(other_tensor != nullptr); //unable to find the linked tensor
  auto other_tensor_leg = other_tensor->getTensorLeg(other_tensor_leg_id);
  other_tensor_leg.resetTensorId(tensor_id);
  other_tensor_leg.resetDimensionId(i);
  other_tensor->resetLeg(other_tensor_leg_id,other_tensor_leg);
 }
 return;
}


void TensorNetwork::updateConnectionsFromInputTensors()
{
 for(auto iter = this->cbegin(); iter != this->cend(); ++iter){
  if(iter->first != 0) updateConnections(iter->first);
 }
 return;
}


void TensorNetwork::invalidateMaxTensorId()
{
 max_tensor_id_ = 0;
 return;
}


void TensorNetwork::invalidateContractionSequence()
{
 split_tensors_.clear();
 split_indices_.clear();
 operations_.clear();
 contraction_seq_.clear();
 contraction_seq_flops_ = 0.0;
 max_intermediate_presence_volume_ = 0.0;
 max_intermediate_volume_ = 0.0;
 max_intermediate_rank_ = 0;
 universal_indexing_ = false;
 return;
}


void TensorNetwork::invalidateTensorOperationList()
{
 split_tensors_.clear();
 split_indices_.clear();
 operations_.clear();
 max_intermediate_presence_volume_ = 0.0;
 max_intermediate_volume_ = 0.0;
 max_intermediate_rank_ = 0;
 universal_indexing_ = false;
 return;
}


double TensorNetwork::determineContractionSequence(ContractionSeqOptimizer & contr_seq_optimizer)
{
 assert(finalized_ != 0); //tensor network must be in finalized state
 if(contraction_seq_.empty()){
  auto intermediate_num_begin = this->getMaxTensorId() + 1;
  auto intermediate_num_generator = [intermediate_num_begin]() mutable {return intermediate_num_begin++;};
  contraction_seq_flops_ = contr_seq_optimizer.determineContractionSequence(*this,contraction_seq_,intermediate_num_generator);
  max_intermediate_presence_volume_ = 0.0;
  max_intermediate_volume_ = 0.0;
  max_intermediate_rank_ = 0;
 }
 return contraction_seq_flops_;
}


double TensorNetwork::determineContractionSequence(const std::string & contr_seq_opt_name)
{
 auto iter = optimizers.find(contr_seq_opt_name);
 if(iter == optimizers.end()){ //not cached
  auto & optimizer_factory = *(ContractionSeqOptimizerFactory::get());
  auto optimizer = optimizer_factory.createContractionSeqOptimizer(contr_seq_opt_name);
  if(optimizer){
   auto res = optimizers.emplace(std::make_pair(contr_seq_opt_name,
                                 std::shared_ptr<ContractionSeqOptimizer>(std::move(optimizer))));
   assert(res.second);
   iter = res.first;
  }else{
   std::cout << "#ERROR(TensorNetwork::determineContractionSequence): Invalid request: " <<
    "Tensor contraction sequence optimizer " << contr_seq_opt_name << " has not been registered before!" << std::endl;
   assert(false);
  }
 }
 return determineContractionSequence(*(iter->second));
}


void TensorNetwork::importContractionSequence(const std::list<ContrTriple> & contr_sequence,
                                              double fma_flops)
{
 assert(finalized_ != 0); //tensor network must be in finalized state
 contraction_seq_.clear();
 contraction_seq_ = contr_sequence;
 contraction_seq_flops_ = fma_flops; //flop count may be unknown yet (defaults to zero)
 max_intermediate_presence_volume_ = 0.0; //max cumulative volume of intermediates present at a time
 max_intermediate_volume_ = 0.0; //max intermediate tensor volume is unknown yet
 max_intermediate_rank_ = 0; //max intermediate tensor rank
 return;
}


void TensorNetwork::importContractionSequence(const std::vector<unsigned int> & contr_sequence_content,
                                              double fma_flops)
{
 assert(finalized_ != 0); //tensor network must be in finalized state
 contraction_seq_.clear();
 unpackContractionSequenceFromVector(contraction_seq_,contr_sequence_content);
 contraction_seq_flops_ = fma_flops; //flop count may be unknown yet (defaults to zero)
 max_intermediate_presence_volume_ = 0.0; //max cumulative volume of intermediates present at a time
 max_intermediate_volume_ = 0.0; //max intermediate tensor volume is unknown yet
 max_intermediate_rank_ = 0; //max intermediate tensor rank
 return;
}


const std::list<ContrTriple> & TensorNetwork::exportContractionSequence(double * fma_flops) const
{
 if(fma_flops != nullptr) *fma_flops = contraction_seq_flops_;
 return contraction_seq_;
}


inline IndexSplit splitDimension(std::pair<SpaceId,SubspaceId> space_attr, //helper
                                 DimExtent dim_extent,
                                 std::size_t num_segments)
{
 assert(dim_extent >= num_segments);
 IndexSplit split_info;
 if(space_attr.first == SOME_SPACE){ //anonymous vector space
  std::size_t seg_size = dim_extent/num_segments;
  std::size_t remainder = dim_extent - seg_size * num_segments;
  SubspaceId base = space_attr.second;
  for(std::size_t i = 0; i < num_segments; ++i){
   DimExtent extent = seg_size; if(i < remainder) extent++;
   split_info.emplace_back(std::pair<SubspaceId,DimExtent>{base,extent});
   base += extent;
  }
  assert(base == space_attr.second + dim_extent);
 }else{ //registered named vector space
  assert(false); //`Implement in future
 }
 return split_info;
}


void TensorNetwork::establishUniversalIndexNumeration()
{
 if(universal_indexing_) return;
 std::unordered_map<TensorHashType,std::string> intermediates; //tensor hash --> symbolic tensor intermediate with universal indices
 std::unordered_map<std::string,std::string> index_map; //old index name --> new index name
 std::vector<std::string> tens_operands; //extracted tensor operands
 std::vector<IndexLabel> indices,new_indices; //indices extracted from a tensor
 std::string tensor_name; //tensor name extracted from a tensor
 std::string new_pattern; //new tensor operation index pattern
 bool conjugated = false;
 bool output_tensor_done = false;
 int num_internal_indices = 0;
 //Update index patterns in all tensor operations in reverse order:
 for(auto op_iter = operations_.rbegin(); op_iter != operations_.rend(); ++op_iter){
  auto & op = *(*op_iter); //tensor operation
  const auto num_operands = op.getNumOperands();
  const auto num_operands_out = op.getNumOperandsOut();
  assert(num_operands <= 3 && num_operands_out <= 1); //`Only expecting regular tensor operations so far
  const auto & old_pattern = op.getIndexPattern();
  if(old_pattern.length() > 0){ //index pattern present
   assert(num_operands > 1 && num_operands_out == 1); //presence of index pattern assumes two or more operands
   //std::cout << "#DEBUG(TensorNetwork::establishUniversalIndexNumeration): Old pattern: " << old_pattern << std::endl; //debug
   tens_operands.clear();
   bool success = parse_tensor_network(old_pattern,tens_operands);
   if(success){
    //Process all tensor operands:
    if(tens_operands.size() == num_operands){
     new_pattern.clear();
     //Process the only output tensor operand (#0):
     tensor_name.clear(); indices.clear();
     success = parse_tensor(tens_operands[0],tensor_name,indices,conjugated);
     if(success){
      //std::cout << " New output tensor: " << tens_operands[0] << std::endl; //debug
      const auto & tens0 = *(op.getTensorOperand(0));
      auto tensor_hash = tens0.getTensorHash();
      //Pre-save the output tensor of the tensor network:
      if(!output_tensor_done){
       assert(!conjugated); //output tensor cannot be conjugated
       auto res = intermediates.emplace(std::make_pair(tensor_hash,
                   assemble_symbolic_tensor(tens0.getName(),indices,conjugated)));
       assert(res.second);
       //std::cout << " Saved tensor: " << res.first->second << std::endl; //debug
       output_tensor_done = true;
      }
      //Retreive the intermediate (output) tensor operand in a universal form:
      auto tens_iter = intermediates.find(tensor_hash); assert(tens_iter != intermediates.end());
      //std::cout << " Found intermediate: " << tens_iter->second << std::endl; //debug
      new_pattern += (tens_iter->second + "+="); //append universally indexed output tensor operand to the new index pattern
      //Establish uncontracted index remapping:
      index_map.clear(); tensor_name.clear(); new_indices.clear();
      success = parse_tensor(tens_iter->second,tensor_name,new_indices,conjugated); assert(success);
      //std::cout << " Sizes of indices: " << indices.size() << " " << new_indices.size() << std::endl; //debug
      assert(new_indices.size() == indices.size());
      for(auto it_old = indices.cbegin(),
               it_new = new_indices.cbegin(); it_old != indices.cend(); ++it_old,
                                                                        ++it_new){
       index_map.emplace(std::make_pair(it_old->label,it_new->label));
      }
      //Process input tensor operands:
      int num_contr_indices = 0;
      for(unsigned int op_num = 1; op_num < num_operands; ++op_num){ //`Assumes a single output tensor operand (#0)
       //std::cout << " New input tensor: " << tens_operands[op_num] << std::endl; //debug
       tensor_name.clear(); indices.clear();
       success = parse_tensor(tens_operands[op_num],tensor_name,indices,conjugated);
       if(success){
        const auto & tens = *(op.getTensorOperand(op_num));
        tensor_hash = tens.getTensorHash();
        //Update the numeration of contracted indices with global numbers and remap uncontracted indices:
        num_contr_indices = 0;
        for(auto & index: indices){
         if(index.label[0] == 'c'){ //contracted index requires global shift
          num_contr_indices++;
          auto old_number = std::stoi(index.label.substr(1));
          index.label = ("c" + std::to_string(num_internal_indices + old_number));
         }else if(index.label[0] == 'u'){ //uncontracted indices need remapping
          index.label = index_map[index.label];
         }else{
          std::cout << "#ERROR(exatn::numerics::TensorNetwork::establishUniversalIndexNumeration): "
                    << "Invalid index label encountered: " << index.label << std::endl;
          assert(false);
         }
        }
        const auto symb_tensor = assemble_symbolic_tensor(tens.getName(),indices,conjugated);
        if(isPureIntermediateTensorName(symb_tensor)){
         assert(!conjugated); //intermediate tensors do not appear conjugated
         auto res = intermediates.emplace(std::make_pair(tensor_hash,symb_tensor));
         if(!res.second){
          std::cout << "#ERROR(exatn::numerics::TensorNetwork::establishUniversalIndexNumeration): "
                    << "Intermediate tensor already saved previously: " << symb_tensor << std::endl;
          assert(false);
         }
         //std::cout << " Saved tensor: " << res.first->second << std::endl; //debug
        }
        if(op_num == 1){
         new_pattern += symb_tensor;
        }else if(op_num == 2){
         new_pattern += ("*" + symb_tensor);
        }else{
         assert(false); //`At most three tensor operands are expected so far
        }
       }else{
        std::cout << "#ERROR(exatn::numerics::TensorNetwork::establishUniversalIndexNumeration): "
                  << "Unable to parse tensor operand: " << tens_operands[op_num] << std::endl;
        assert(false);
       }
      }
      num_internal_indices += num_contr_indices;
      op.setIndexPattern(new_pattern);
      //std::cout << " New index pattern: " << new_pattern << std::endl; //debug
     }else{
      std::cout << "#ERROR(exatn::numerics::TensorNetwork::establishUniversalIndexNumeration): "
                << "Unable to parse tensor operand: " << tens_operands[0] << std::endl;
      assert(false);
     }
    }else{
     std::cout << "#ERROR(exatn::numerics::TensorNetwork::establishUniversalIndexNumeration): "
               << "Invalid number of tensor operands (" << tens_operands.size() << " VS " << num_operands
               << ") parsed from: " << old_pattern << ": ";
     for(const auto & operand: tens_operands) std::cout << operand << " ";
     std::cout << std::endl;
     op.printIt();
     assert(false);
    }
   }else{
    std::cout << "#ERROR(exatn::numerics::TensorNetwork::establishUniversalIndexNumeration): "
              << "Unable to parse tensor operation index pattern: " << old_pattern << std::endl;
    assert(false);
   }
  }
 }
 universal_indexing_ = true;
 return;
}


bool TensorNetwork::placeTensor(unsigned int tensor_id,                     //in: tensor id (unique within the tensor network)
                                std::shared_ptr<Tensor> tensor,             //in: appended tensor
                                const std::vector<TensorLeg> & connections, //in: tensor connections (fully specified)
                                bool conjugated,                            //in: complex conjugation flag for the appended tensor
                                bool leg_matching_check)                    //in: tensor leg matching check
{
 if(explicit_output_ == 0){
  std::cout << "#ERROR(TensorNetwork::placeTensor): Invalid request: " <<
   "Appending a tensor via explicit connections to the tensor network that is missing a full output tensor!" << std::endl;
  return false;
 }
 if(finalized_ != 0){
  std::cout << "#ERROR(TensorNetwork::placeTensor): Invalid request: " <<
   "Appending a tensor via explicit connections to the tensor network that has been finalized!" << std::endl;
  return false;
 }
 if(tensor_id == 0){
  std::cout << "#ERROR(TensorNetwork::placeTensor): Invalid request: " <<
   "Attempt to append an output tensor (id = 0) to a tensor network with an explicit output tensor!" << std::endl;
  return false;
 }
 //Check the validity of new connections:
 if(leg_matching_check){
  unsigned int mode = 0;
  for(const auto & leg: connections){
   const auto * tensconn = this->getTensorConn(leg.getTensorId());
   if(tensconn != nullptr){ //connected tensor is already in the tensor network
    const auto & tens_legs = tensconn->getTensorLegs();
    const auto & tens_leg = tens_legs[leg.getDimensionId()];
    if(tens_leg.getTensorId() != tensor_id || tens_leg.getDimensionId() != mode){
     std::cout << "#ERROR(TensorNetwork::placeTensor): Invalid argument: Connections are invalid: "
               << "Failed input leg: "; leg.printIt(); std::cout << std::endl;
     return false;
    }
   }
   ++mode;
  }
 }
 //Append the tensor to the tensor network:
 auto res = emplaceTensorConnDirect(true,
                                    tensor_id,
                                    tensor,tensor_id,connections,conjugated);
 if(!res){
  std::cout << "#ERROR(TensorNetwork::placeTensor): Invalid request: " <<
   "A tensor with id " << tensor_id << " already exists in the tensor network!" << std::endl;
  return false;
 }
 return true;
}


bool TensorNetwork::appendTensor(unsigned int tensor_id,
                                 std::shared_ptr<Tensor> tensor,
                                 const std::vector<std::pair<unsigned int, unsigned int>> & pairing,
                                 const std::vector<LegDirection> & leg_dir,
                                 bool conjugated)
{
 if(explicit_output_ != 0 && finalized_ == 0){
  std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid request: " <<
   "Appending a tensor via implicit pairing with the output tensor, but the tensor network is not finalized!" << std::endl;
  return false;
 }
 if(tensor_id == 0){
  std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid request: " <<
   "Attempt to append an output tensor (id = 0) to a finalized tensor network!" << std::endl;
  return false;
 }
 //Reset the output tensor to a new one:
 this->resetOutputTensor();
 //Check validity of leg pairing:
 auto tensor_rank = tensor->getRank();
 bool dir_present = (leg_dir.size() > 0);
 if(dir_present && leg_dir.size() != tensor_rank){
  std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid argument: Incomplete vector of leg directions!" << std::endl;
  return false;
 }
 auto * output_tensor = this->getTensorConn(0);
 assert(output_tensor != nullptr); //output tensor must be present
 auto output_rank = output_tensor->getNumLegs();
 if(output_rank > 0 && tensor_rank > 0){
  int ouf[output_rank] = {0};
  int tef[tensor_rank] = {0};
  for(const auto & link: pairing){
   if(link.first >= output_rank || link.second >= tensor_rank){
    std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid argument: Invalid leg pairing!" << std::endl;
    return false;
   }
   if(ouf[link.first]++ != 0){
    std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid argument: Pairing: Repeated output leg!" << std::endl;
    return false;
   }
   if(tef[link.second]++ != 0){
    std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid argument: Pairing: Repeated new tensor leg!" << std::endl;
    return false;
   }
  }
 }else{
  if(pairing.size() > 0){
   std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid argument: Pairing: Pairing on a scalar tensor!" << std::endl;
   return false;
  }
 }
 //Pair legs of the new tensor with the input tensors of the tensor network:
 if(tensor_rank > 0){ //true tensor
  std::vector<TensorLeg> new_tensor_legs(tensor_rank,TensorLeg(0,0)); //placeholders for legs
  if(pairing.size() > 0){
   std::vector<unsigned int> matched_output_legs(pairing.size(),0);
   unsigned int mode = 0;
   for(const auto & link: pairing){
    const auto & output_tensor_leg_id = link.first;
    const auto & tensor_leg_id = link.second;
    auto output_tensor_leg = output_tensor->getTensorLeg(output_tensor_leg_id);
    const auto input_tensor_id = output_tensor_leg.getTensorId();
    const auto input_tensor_leg_id = output_tensor_leg.getDimensionId();
    auto * input_tensor = this->getTensorConn(input_tensor_id);
    assert(input_tensor != nullptr);
    auto input_tensor_leg = input_tensor->getTensorLeg(input_tensor_leg_id);
    input_tensor_leg.resetTensorId(tensor_id);
    input_tensor_leg.resetDimensionId(tensor_leg_id);
    input_tensor->resetLeg(input_tensor_leg_id,input_tensor_leg);
    new_tensor_legs[tensor_leg_id].resetTensorId(input_tensor_id);
    new_tensor_legs[tensor_leg_id].resetDimensionId(input_tensor_leg_id);
    if(dir_present){
     new_tensor_legs[tensor_leg_id].resetDirection(leg_dir[tensor_leg_id]);
     if(input_tensor_leg.getDirection() != reverseLegDirection(new_tensor_legs[tensor_leg_id].getDirection())){
      std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid argument: Leg directions: Pairing leg direction mismatch!" << std::endl;
      return false;
     }
    }else{
     new_tensor_legs[tensor_leg_id].resetDirection(reverseLegDirection(input_tensor_leg.getDirection()));
    }
    matched_output_legs[mode++] = output_tensor_leg_id;
   }
   //Delete matched legs from the output tensor:
   output_tensor->deleteLegs(matched_output_legs);
   updateConnections(0); //update tensor network connections due to deletion of the matched output tensor legs
  }
  //Append unpaired legs of the new tensor to the output tensor of the network:
  output_rank = output_tensor->getNumLegs();
  unsigned int mode = 0;
  for(auto & leg: new_tensor_legs){
   if(leg.getTensorId() == 0){ //unpaired tensor leg
    LegDirection dir = LegDirection::UNDIRECT;
    if(dir_present) dir = leg_dir[mode];
    leg.resetDimensionId(output_rank);
    leg.resetDirection(dir);
    output_tensor->appendLeg(tensor->getDimSpaceAttr(mode),tensor->getDimExtent(mode),
                             TensorLeg(tensor_id,mode,reverseLegDirection(dir)));
    output_rank = output_tensor->getNumLegs();
   }
   ++mode;
  }
  auto res = emplaceTensorConnDirect(true,
                                     tensor_id,
                                     tensor,tensor_id,new_tensor_legs,conjugated);
  if(!res){
   std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid request: " <<
    "A tensor with id " << tensor_id << " already exists in the tensor network!" << std::endl;
   return false;
  }
 }else{ //scalar tensor
  auto res = emplaceTensorConnDirect(true,
                                     tensor_id,
                                     tensor,tensor_id,std::vector<TensorLeg>{},conjugated);
  if(!res){
   std::cout << "#ERROR(TensorNetwork::appendTensor): Invalid request: " <<
    "A tensor with id " << tensor_id << " already exists in the tensor network!" << std::endl;
   return false;
  }
 }
 invalidateContractionSequence(); //invalidate previously cached tensor contraction sequence
 finalized_ = 1; //implicit leg pairing always keeps the tensor network in a finalized state
 return true;
}


bool TensorNetwork::appendTensor(std::shared_ptr<Tensor> tensor,
                                 const std::vector<std::pair<unsigned int, unsigned int>> & pairing,
                                 const std::vector<LegDirection> & leg_dir,
                                 bool conjugated)
{
 return appendTensor(getMaxTensorId()+1,tensor,pairing,leg_dir,conjugated);
}


bool TensorNetwork::appendTensorGate(unsigned int tensor_id,
                                     std::shared_ptr<Tensor> tensor,
                                     const std::vector<unsigned int> & pairing,
                                     bool conjugated)
{
 if(finalized_ == 0){
  std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid request: " <<
   "Appending a tensor gate to an unfinalized tensor network is forbidden!" << std::endl;
  return false;
 }
 if(tensor_id == 0){
  std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid request: " <<
   "Tensor 0 (output tensor) must already be present in the tensor network!" << std::endl;
  return false;
 }
 //Reset the output tensor to a new one:
 this->resetOutputTensor();
 //Check validity of leg pairing:
 auto * output_tensor = this->getTensorConn(0);
 assert(output_tensor != nullptr); //output tensor must be present
 auto output_rank = output_tensor->getNumLegs();
 auto tensor_rank = tensor->getRank();
 if((tensor_rank % 2) != 0){
  std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid argument: Odd-rank tensors are not allowed as gates!" << std::endl;
  return false;
 }
 if(tensor_rank != pairing.size() * 2){
  std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid argument: Wrong size of the leg pairing vector!" << std::endl;
  return false;
 }
 if(tensor_rank > output_rank * 2){
  std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid argument: Tensor network does not have enough open legs!" << std::endl;
  return false;
 }
 if(output_rank > 0){
  char inds[output_rank] = {0};
  for(const auto & leg_id: pairing){
   if(leg_id >= output_rank){
    std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid argument: Invalid content of the pairing vector!" << std::endl;
    return false;
   }
   if(inds[leg_id]++ != 0){
    std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid argument: Invalid content of the pairing vector!" << std::endl;
    return false;
   }
  }
 }
 //Pair legs of the new tensor with the input tensors of the tensor network:
 if(tensor_rank > 0){
  std::vector<TensorLeg> new_tensor_legs(tensor_rank,TensorLeg(0,0)); //placeholders for legs
  unsigned int paired_leg_id = 0;
  unsigned int unpaired_leg_id = tensor_rank / 2;
  if(conjugated) std::swap(paired_leg_id,unpaired_leg_id);
  for(const auto & output_tensor_leg_id: pairing){
   auto output_tensor_leg = output_tensor->getTensorLeg(output_tensor_leg_id);
   //Relink the input tensor with the new tensor:
   const auto input_tensor_id = output_tensor_leg.getTensorId();
   const auto input_tensor_leg_id = output_tensor_leg.getDimensionId();
   auto * input_tensor = this->getTensorConn(input_tensor_id);
   assert(input_tensor != nullptr);
   auto input_tensor_leg = input_tensor->getTensorLeg(input_tensor_leg_id);
   input_tensor_leg.resetTensorId(tensor_id);
   input_tensor_leg.resetDimensionId(paired_leg_id);
   input_tensor->resetLeg(input_tensor_leg_id,input_tensor_leg);
   new_tensor_legs[paired_leg_id].resetTensorId(input_tensor_id);
   new_tensor_legs[paired_leg_id].resetDimensionId(input_tensor_leg_id);
   new_tensor_legs[paired_leg_id].resetDirection(reverseLegDirection(input_tensor_leg.getDirection()));
   //Relink the output tensor leg with the new tensor:
   output_tensor_leg.resetTensorId(tensor_id);
   output_tensor_leg.resetDimensionId(unpaired_leg_id);
   output_tensor->resetLeg(output_tensor_leg_id,output_tensor_leg);
   new_tensor_legs[unpaired_leg_id].resetTensorId(0);
   new_tensor_legs[unpaired_leg_id].resetDimensionId(output_tensor_leg_id);
   new_tensor_legs[unpaired_leg_id].resetDirection(reverseLegDirection(output_tensor_leg.getDirection()));
   ++paired_leg_id; ++unpaired_leg_id;
  }
  auto res = emplaceTensorConnDirect(true,
                                     tensor_id,
                                     tensor,tensor_id,new_tensor_legs,conjugated);
  if(!res){
   std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid request: " <<
    "A tensor with id " << tensor_id << " already exists in the tensor network!" << std::endl;
   return false;
  }
 }else{ //scalar tensor
  auto res = emplaceTensorConnDirect(true,
                                     tensor_id,
                                     tensor,tensor_id,std::vector<TensorLeg>{},conjugated);
  if(!res){
   std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid request: " <<
    "A tensor with id " << tensor_id << " already exists in the tensor network!" << std::endl;
   return false;
  }
 }
 invalidateContractionSequence(); //invalidate previously cached tensor contraction sequence
 finalized_ = 1; //implicit leg pairing always keeps the tensor network in a finalized state
 return true;
}


bool TensorNetwork::appendTensorGate(std::shared_ptr<Tensor> tensor,
                                     const std::vector<unsigned int> & pairing,
                                     bool conjugated)
{
 return appendTensorGate(getMaxTensorId()+1,tensor,pairing,conjugated);
}


bool TensorNetwork::appendTensorGateGeneral(unsigned int tensor_id,
                                            std::shared_ptr<Tensor> tensor,
                                            const std::vector<std::pair<unsigned int,
                                                                        std::pair<unsigned int,
                                                                                  unsigned int>>> & pairing,
                                            bool conjugated)
{
 if(finalized_ == 0){
  std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid request: " <<
   "Appending a tensor gate to an unfinalized tensor network is forbidden!" << std::endl;
  return false;
 }
 if(tensor_id == 0){
  std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid request: " <<
   "Tensor 0 (output tensor) must already be present in the tensor network!" << std::endl;
  return false;
 }
 //Reset the output tensor to a new one:
 this->resetOutputTensor();
 //Check validity of leg pairing:
 auto * output_tensor = this->getTensorConn(0);
 assert(output_tensor != nullptr); //output tensor must be present
 auto output_rank = output_tensor->getNumLegs();
 auto tensor_rank = tensor->getRank();
 if((tensor_rank % 2) != 0){
  std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid argument: Odd-rank tensors are not allowed as gates!" << std::endl;
  return false;
 }
 if(tensor_rank != pairing.size() * 2){
  std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid argument: Wrong size of the leg pairing vector!" << std::endl;
  return false;
 }
 if(tensor_rank > output_rank * 2){
  std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid argument: Tensor network does not have enough open legs!" << std::endl;
  return false;
 }
 if(output_rank > 0){
  char inds[output_rank] = {0};
  for(const auto & leg_match: pairing){
   if(leg_match.first >= output_rank){
    std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid argument: Invalid content of the pairing vector!" << std::endl;
    return false;
   }
   if(inds[leg_match.first]++ != 0){
    std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid argument: Invalid content of the pairing vector!" << std::endl;
    return false;
   }
  }
 }
 if(tensor_rank > 0){
  char inds[tensor_rank] = {0};
  for(const auto & leg_match: pairing){
   if(leg_match.second.first >= tensor_rank){
    std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid argument: Invalid content of the pairing vector!" << std::endl;
    return false;
   }
   if(inds[leg_match.second.first]++ != 0){
    std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid argument: Invalid content of the pairing vector!" << std::endl;
    return false;
   }
   if(leg_match.second.second >= tensor_rank){
    std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid argument: Invalid content of the pairing vector!" << std::endl;
    return false;
   }
   if(inds[leg_match.second.second]++ != 0){
    std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid argument: Invalid content of the pairing vector!" << std::endl;
    return false;
   }
  }
 }
 //Pair legs of the new tensor with the input tensors of the tensor network:
 if(tensor_rank > 0){
  std::vector<TensorLeg> new_tensor_legs(tensor_rank,TensorLeg(0,0)); //placeholders for legs
  for(const auto & tensor_leg_match: pairing){
   unsigned int output_tensor_leg_id = tensor_leg_match.first;
   auto output_tensor_leg = output_tensor->getTensorLeg(output_tensor_leg_id);
   unsigned int paired_leg_id = tensor_leg_match.second.first;
   unsigned int unpaired_leg_id = tensor_leg_match.second.second;
   if(conjugated) std::swap(paired_leg_id,unpaired_leg_id);
   //Relink the input tensor with the new tensor:
   const auto input_tensor_id = output_tensor_leg.getTensorId();
   const auto input_tensor_leg_id = output_tensor_leg.getDimensionId();
   auto * input_tensor = this->getTensorConn(input_tensor_id);
   assert(input_tensor != nullptr);
   auto input_tensor_leg = input_tensor->getTensorLeg(input_tensor_leg_id);
   input_tensor_leg.resetTensorId(tensor_id);
   input_tensor_leg.resetDimensionId(paired_leg_id);
   input_tensor->resetLeg(input_tensor_leg_id,input_tensor_leg);
   new_tensor_legs[paired_leg_id].resetTensorId(input_tensor_id);
   new_tensor_legs[paired_leg_id].resetDimensionId(input_tensor_leg_id);
   new_tensor_legs[paired_leg_id].resetDirection(reverseLegDirection(input_tensor_leg.getDirection()));
   //Relink the output tensor leg with the new tensor:
   output_tensor_leg.resetTensorId(tensor_id);
   output_tensor_leg.resetDimensionId(unpaired_leg_id);
   output_tensor->resetLeg(output_tensor_leg_id,output_tensor_leg);
   new_tensor_legs[unpaired_leg_id].resetTensorId(0);
   new_tensor_legs[unpaired_leg_id].resetDimensionId(output_tensor_leg_id);
   new_tensor_legs[unpaired_leg_id].resetDirection(reverseLegDirection(output_tensor_leg.getDirection()));
  }
  auto res = emplaceTensorConnDirect(true,
                                     tensor_id,
                                     tensor,tensor_id,new_tensor_legs,conjugated);
  if(!res){
   std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid request: " <<
    "A tensor with id " << tensor_id << " already exists in the tensor network!" << std::endl;
   return false;
  }
 }else{ //scalar tensor
  auto res = emplaceTensorConnDirect(true,
                                     tensor_id,
                                     tensor,tensor_id,std::vector<TensorLeg>{},conjugated);
  if(!res){
   std::cout << "#ERROR(TensorNetwork::appendTensorGate): Invalid request: " <<
    "A tensor with id " << tensor_id << " already exists in the tensor network!" << std::endl;
   return false;
  }
 }
 invalidateContractionSequence(); //invalidate previously cached tensor contraction sequence
 finalized_ = 1; //implicit leg pairing always keeps the tensor network in a finalized state
 return true;
}


bool TensorNetwork::appendTensorGateGeneral(std::shared_ptr<Tensor> tensor,
                                            const std::vector<std::pair<unsigned int,
                                                                        std::pair<unsigned int,
                                                                                  unsigned int>>> & pairing,
                                            bool conjugated)
{
 return appendTensorGateGeneral(getMaxTensorId()+1,tensor,pairing,conjugated);
}


bool TensorNetwork::appendTensorNetwork(TensorNetwork && network,                                           //in: appended tensor network
                                        const std::vector<std::pair<unsigned int, unsigned int>> & pairing) //in: leg pairing: output tensor mode (primary) -> output tensor mode (appended)
{
 if(!((*this).isFinalized()) || !(network.isFinalized())){
  std::cout << "#ERROR(TensorNetwork::appendTensorNetwork): Invalid request: " <<
   "Either primary or appended tensor network is not finalized!" << std::endl;
  return false;
 }
 //Reset the output tensor to a new one:
 this->resetOutputTensor();
 network.resetOutputTensor();
 //Check validity of leg pairing:
 auto * output0 = this->getTensorConn(0);
 assert(output0 != nullptr);
 auto output0_rank = output0->getNumLegs();
 auto * output1 = network.getTensorConn(0);
 assert(output1 != nullptr);
 auto output1_rank = output1->getNumLegs();
 if(output0_rank > 0 && output1_rank > 0){
  int ou0[output0_rank] = {0};
  int ou1[output1_rank] = {0};
  for(const auto & link: pairing){
   if(link.first >= output0_rank || link.second >= output1_rank){
    std::cout << "#ERROR(TensorNetwork::appendTensorNetwork): Invalid argument: Pairing: Out of bounds!" << std::endl;
    return false;
   }
   if(ou0[link.first]++ != 0){
    std::cout << "#ERROR(TensorNetwork::appendTensorNetwork): Invalid argument: Pairing: Repeated primary output leg!" << std::endl;
    return false;
   }
   if(ou1[link.second]++ != 0){
    std::cout << "#ERROR(TensorNetwork::appendTensorNetwork): Invalid argument: Pairing: Repeated secondary output leg!" << std::endl;
    return false;
   }
  }
 }else{
  if(pairing.size() > 0){
   std::cout << "#ERROR(TensorNetwork::appendTensorNetwork): Invalid argument: Pairing: Non-trivial pairing on scalar networks!" << std::endl;
   return false;
  }
 }
 //Shift input tensor numeration in all internal legs of the appended tensor network:
 auto max_tensor_id = this->getMaxTensorId(); assert(max_tensor_id > 0);
 for(auto tensor_conn_iter = network.begin(); tensor_conn_iter != network.end(); ++tensor_conn_iter){
  if(tensor_conn_iter->first != 0){
   auto & tensor_conn = tensor_conn_iter->second;
   const auto tensor_conn_rank = tensor_conn.getNumLegs();
   for(unsigned int i = 0; i < tensor_conn_rank; ++i){
    TensorLeg new_leg = tensor_conn.getTensorLeg(i);
    const auto conn_tensor_id = new_leg.getTensorId();
    if(conn_tensor_id != 0){
     new_leg.resetTensorId(conn_tensor_id+max_tensor_id);
     tensor_conn.resetLeg(i,new_leg);
    }
   }
  }
 }
 //Pair output legs of the primary tensor network with output legs of the appended (secondary) tensor network:
 if(pairing.size() > 0){
  for(const auto & link: pairing){
   const auto & output0_leg_id = link.first;
   const auto & output1_leg_id = link.second;
   const auto & output0_leg = output0->getTensorLeg(output0_leg_id);
   const auto & output1_leg = output1->getTensorLeg(output1_leg_id);
   const auto input0_id = output0_leg.getTensorId();
   const auto input0_leg_id = output0_leg.getDimensionId();
   const auto input1_id = output1_leg.getTensorId();
   const auto input1_leg_id = output1_leg.getDimensionId();
   auto * input0 = this->getTensorConn(input0_id);
   assert(input0 != nullptr);
   auto * input1 = network.getTensorConn(input1_id);
   assert(input1 != nullptr);
   auto input0_leg = input0->getTensorLeg(input0_leg_id);
   input0_leg.resetTensorId(input1_id+max_tensor_id); //shift other network's tensor ids
   input0_leg.resetDimensionId(input1_leg_id);
   input0->resetLeg(input0_leg_id,input0_leg);
   auto input1_leg = input1->getTensorLeg(input1_leg_id);
   input1_leg.resetTensorId(input0_id);
   input1_leg.resetDimensionId(input0_leg_id);
   input1->resetLeg(input1_leg_id,input1_leg);
  }
  //Delete matched legs from both output tensors:
  std::vector<unsigned int> matched_output_legs(pairing.size(),0);
  for(unsigned int i = 0; i < pairing.size(); ++i) matched_output_legs[i] = pairing[i].first;
  output0->deleteLegs(matched_output_legs);
  this->updateConnections(0);
  for(unsigned int i = 0; i < pairing.size(); ++i) matched_output_legs[i] = pairing[i].second;
  output1->deleteLegs(matched_output_legs);
  network.updateConnections(0);
 }
 //Append unmatched legs of the output tensor from the appended tensor network to the output tensor from the primary tensor network:
 output0_rank = output0->getNumLegs();
 output1_rank = output1->getNumLegs();
 for(unsigned int i = 0; i < output1_rank; ++i){
  TensorLeg out1_leg(output1->getTensorLeg(i));
  out1_leg.resetTensorId(out1_leg.getTensorId()+max_tensor_id);
  output0->appendLeg(output1->getDimSpaceAttr(i),
                     output1->getDimExtent(i),
                     out1_leg);
 }
 output0_rank = output0->getNumLegs();
 //Append all input tensors from the appended tensor network into the primary tensor network:
 auto tensors = network.getTensorConnAll();
 for(auto & tensor: tensors){
  unsigned int tensor_id = tensor->getTensorId() + max_tensor_id;
  auto res = emplaceTensorConn(false,tensor_id,*tensor);
  if(!res){
   std::cout << "#ERROR(exatn::numerics::TensorNetwork::appendTensorNetwork): Tensor id already in use!" << std::endl;
   return false;
  }
 }
 this->updateConnections(0); //update connections in just appended input tensors
 invalidateContractionSequence(); //invalidate previously cached tensor contraction sequence
 finalized_ = 1; //implicit leg pairing always keeps the primary tensor network in a finalized state
 return true;
}


bool TensorNetwork::appendTensorNetworkGate(TensorNetwork && network,
                                            const std::vector<unsigned int> & pairing)
{
 if(!((*this).isFinalized()) || !(network.isFinalized())){
  std::cout << "#ERROR(TensorNetwork::appendTensorNetworkGate): Invalid request: " <<
   "Either primary or appended tensor network is not finalized!" << std::endl;
  return false;
 }
 //Reset the output tensor to a new one:
 this->resetOutputTensor();
 network.resetOutputTensor();
 //Check validity of leg pairing:
 auto * output0 = this->getTensorConn(0);
 assert(output0 != nullptr);
 auto output0_rank = output0->getNumLegs();
 auto * output1 = network.getTensorConn(0);
 assert(output1 != nullptr);
 auto output1_rank = output1->getNumLegs();
 if(output1_rank % 2 != 0){
  std::cout << "#ERROR(TensorNetwork::appendTensorNetworkGate): Invalid argument: Odd-rank tensor networks are not allowed as gates!"
            << std::endl;
  return false;
 }
 if(output1_rank != pairing.size() * 2){
  std::cout << "#ERROR(TensorNetwork::appendTensorNetworkGate): Invalid argument: Wrong size of the leg pairing vector!"
            << std::endl;
  return false;
 }
 if(output1_rank > output0_rank * 2){
  std::cout << "#ERROR(TensorNetwork::appendTensorNetworkGate): Invalid argument: Primary tensor network does not have enough open legs!"
            << std::endl;
  return false;
 }
 if(output0_rank > 0){
  char inds[output0_rank] = {0};
  for(const auto & leg_id: pairing){
   if(leg_id >= output0_rank){
    std::cout << "#ERROR(TensorNetwork::appendTensorNetworkGate): Invalid argument: Invalid content of the pairing vector!" << std::endl;
    return false;
   }
   if(inds[leg_id]++ != 0){
    std::cout << "#ERROR(TensorNetwork::appendTensorNetworkGate): Invalid argument: Invalid content of the pairing vector!" << std::endl;
    return false;
   }
  }
 }
 //Shift input tensor numeration in all internal legs of the appended tensor network:
 auto max_tensor_id = this->getMaxTensorId(); assert(max_tensor_id > 0);
 for(auto tensor_conn_iter = network.begin(); tensor_conn_iter != network.end(); ++tensor_conn_iter){
  if(tensor_conn_iter->first != 0){
   auto & tensor_conn = tensor_conn_iter->second;
   const auto tensor_conn_rank = tensor_conn.getNumLegs();
   for(unsigned int i = 0; i < tensor_conn_rank; ++i){
    TensorLeg new_leg = tensor_conn.getTensorLeg(i);
    const auto conn_tensor_id = new_leg.getTensorId();
    if(conn_tensor_id != 0){
     new_leg.resetTensorId(conn_tensor_id+max_tensor_id);
     tensor_conn.resetLeg(i,new_leg);
    }
   }
  }
 }
 //Pair output legs of the primary tensor network with the output legs of the appended tensor network:
 if(pairing.size() > 0){
  unsigned int output1_leg_id = 0;
  unsigned int output1_replace_leg_id = output1_rank / 2;
  for(const auto & output0_leg_id: pairing){
   const auto & output0_leg = output0->getTensorLeg(output0_leg_id);
   const auto & output1_leg = output1->getTensorLeg(output1_leg_id);
   const auto input0_id = output0_leg.getTensorId();
   const auto input0_leg_id = output0_leg.getDimensionId();
   const auto input1_id = output1_leg.getTensorId();
   const auto input1_leg_id = output1_leg.getDimensionId();
   auto * input0 = this->getTensorConn(input0_id);
   assert(input0 != nullptr);
   auto * input1 = network.getTensorConn(input1_id);
   assert(input1 != nullptr);
   auto input0_leg = input0->getTensorLeg(input0_leg_id);
   input0_leg.resetTensorId(input1_id+max_tensor_id); //shift other network's tensor ids
   input0_leg.resetDimensionId(input1_leg_id);
   input0->resetLeg(input0_leg_id,input0_leg);
   auto input1_leg = input1->getTensorLeg(input1_leg_id);
   input1_leg.resetTensorId(input0_id);
   input1_leg.resetDimensionId(input0_leg_id);
   input1->resetLeg(input1_leg_id,input1_leg);
   TensorLeg output1_replace_leg(output1->getTensorLeg(output1_replace_leg_id));
   output1_replace_leg.resetTensorId(output1_replace_leg.getTensorId()+max_tensor_id);
   output0->resetLeg(output0_leg_id,output1_replace_leg);
   ++output1_leg_id; ++output1_replace_leg_id;
  }
  //Delete matched legs in the output tensor of the appended tensor network:
  std::vector<unsigned int> matched_legs(pairing.size(),0);
  for(unsigned int i = 0; i < pairing.size(); ++i) matched_legs[i] = i; //first half has been matched
  output1->deleteLegs(matched_legs);
  network.updateConnections(0);
 }
 //Append all input tensors from the appended tensor network into the primary tensor network:
 auto tensors = network.getTensorConnAll();
 for(auto & tensor: tensors){
  unsigned int tensor_id = tensor->getTensorId() + max_tensor_id;
  auto res = emplaceTensorConn(false,tensor_id,*tensor);
  if(!res){
   std::cout << "#ERROR(exatn::numerics::TensorNetwork::appendTensorNetworkGate): Tensor id already in use!" << std::endl;
   return false;
  }
 }
 this->updateConnections(0); //update connections in just appended input tensors
 invalidateContractionSequence(); //invalidate previously cached tensor contraction sequence
 finalized_ = 1; //implicit leg pairing always keeps the primary tensor network in a finalized state
 return true;
}


bool TensorNetwork::reorderOutputModes(const std::vector<unsigned int> & order)
{
 if(finalized_ == 0){
  std::cout << "#ERROR(TensorNetwork::reorderOutputModes): Invalid request: " <<
   "Reordering modes in the output tensor of an unfinalized tensor network is forbidden!" << std::endl;
  return false;
 }
 const auto output_tensor_rank = this->getTensorConn(0)->getNumLegs();
 if(order.size() != output_tensor_rank){
  std::cout << "#ERROR(TensorNetwork::reorderOutputModes): Invalid argument: Dimension order: Wrong length: "
            << order.size() << " versus " << output_tensor_rank << std::endl;
  return false;
 }
 if(output_tensor_rank > 0){
  resetOutputTensor(order);
  updateConnections(0);
 }
 return true;
}


bool TensorNetwork::deleteTensor(unsigned int tensor_id)
{
 if(tensor_id == 0){
  std::cout << "#ERROR(TensorNetwork::deleteTensor): Invalid request: " <<
   "Deleting the output tensor of the tensor network is forbidden!" << std::endl;
  return false;
 }
 if(finalized_ == 0){
  std::cout << "#ERROR(TensorNetwork::deleteTensor): Invalid request: " <<
   "Deleting a tensor from an unfinalized tensor network is forbidden!" << std::endl;
  return false;
 }
 //Reset the output tensor to a new one:
 this->resetOutputTensor();
 //Append the released legs from the deleted tensor to the output tensor:
 auto * tensor = this->getTensorConn(tensor_id);
 if(tensor == nullptr){
  std::cout << "#ERROR(TensorNetwork::deleteTensor): Invalid request: " <<
   "Tensor with id " << tensor_id << " is not found in the tensor network!" << std::endl;
  return false;
 }
 //Reconnect the tensors connected to the deleted tensor to the output tensor:
 auto tensor_rank = tensor->getNumLegs();
 if(tensor_rank > 0){
  auto * output_tensor = this->getTensorConn(0);
  assert(output_tensor != nullptr);
  auto output_tensor_rank = output_tensor->getNumLegs();
  //Reconnect input tensors:
  std::vector<unsigned int> orphaned_legs;
  const auto & legs = tensor->getTensorLegs();
  for(const auto & leg: legs){
   const auto other_tensor_id = leg.getTensorId();
   const auto other_tensor_leg_id = leg.getDimensionId();
   if(other_tensor_id != 0){ //connections to the output tensor are ingored (they will disappear)
    auto * other_tensor = this->getTensorConn(other_tensor_id);
    assert(other_tensor != nullptr);
    auto other_tensor_leg = other_tensor->getTensorLeg(other_tensor_leg_id);
    other_tensor_leg.resetTensorId(0);
    other_tensor_leg.resetDimensionId(output_tensor_rank);
    other_tensor->resetLeg(other_tensor_leg_id,other_tensor_leg);
    output_tensor->appendLeg(other_tensor->getDimSpaceAttr(other_tensor_leg_id),
                             other_tensor->getDimExtent(other_tensor_leg_id),
                             TensorLeg(
                              other_tensor_id,
                              other_tensor_leg_id,
                              reverseLegDirection(other_tensor_leg.getDirection())
                             )
                            );
    output_tensor_rank = output_tensor->getNumLegs();
   }else{ //orphaned leg (former connection to the output tensor)
    orphaned_legs.emplace_back(other_tensor_leg_id);
   }
  }
  //Delete orphaned legs of the output tensor:
  if(orphaned_legs.size() > 0){
   output_tensor->deleteLegs(orphaned_legs);
   this->updateConnections(0);
  }
 }
 //Delete the tensor from the network:
 tensor = nullptr;
 auto erased = eraseTensorConn(tensor_id); assert(erased);
 invalidateContractionSequence(); //invalidate previously cached tensor contraction sequence
 return true;
}


bool TensorNetwork::differentiateTensor(unsigned int tensor_id, bool * deltas_appended)
{
 if(deltas_appended != nullptr) *deltas_appended = false;
 if(tensor_id == 0){
  std::cout << "#ERROR(TensorNetwork::differentiateTensor): Invalid request: " <<
   "Differentiating against the output tensor of the tensor network is forbidden!" << std::endl;
  return false;
 }
 if(finalized_ == 0){
  std::cout << "#ERROR(TensorNetwork::differentiateTensor): Invalid request: " <<
   "Differentiation of an unfinalized tensor network is forbidden!" << std::endl;
  return false;
 }
 //Reset the output tensor to a new one:
 this->resetOutputTensor();
 //Append the released legs from the deleted tensor to the output tensor:
 auto * tensor = this->getTensorConn(tensor_id);
 if(tensor == nullptr){
  std::cout << "#ERROR(TensorNetwork::differentiateTensor): Invalid request: " <<
   "Tensor with id " << tensor_id << " is not found in the tensor network!" << std::endl;
  return false;
 }
 //Reconnect the tensors connected to the deleted tensor to the output tensor:
 auto tensor_rank = tensor->getNumLegs();
 if(tensor_rank > 0){
  auto * output_tensor = this->getTensorConn(0);
  assert(output_tensor != nullptr);
  auto output_tensor_rank = output_tensor->getNumLegs();
  //Reconnect input tensors:
  std::vector<unsigned int> orphaned_legs;
  const auto & legs = tensor->getTensorLegs();
  for(const auto & leg: legs){
   const auto other_tensor_id = leg.getTensorId();
   const auto other_tensor_leg_id = leg.getDimensionId();
   if(other_tensor_id != 0){ //connections to the output tensor will require addition of delta tensors
    auto * other_tensor = this->getTensorConn(other_tensor_id);
    assert(other_tensor != nullptr);
    auto other_tensor_leg = other_tensor->getTensorLeg(other_tensor_leg_id);
    other_tensor_leg.resetTensorId(0);
    other_tensor_leg.resetDimensionId(output_tensor_rank);
    other_tensor->resetLeg(other_tensor_leg_id,other_tensor_leg);
    output_tensor->appendLeg(other_tensor->getDimSpaceAttr(other_tensor_leg_id),
                             other_tensor->getDimExtent(other_tensor_leg_id),
                             TensorLeg(
                              other_tensor_id,
                              other_tensor_leg_id,
                              reverseLegDirection(other_tensor_leg.getDirection())
                             )
                            );
    output_tensor_rank = output_tensor->getNumLegs();
   }else{ //orphaned leg (former connection to the output tensor)
    orphaned_legs.emplace_back(other_tensor_leg_id);
   }
  }
  //Append delta tensors for orphaned legs of the output tensor:
  if(orphaned_legs.size() > 0){
   for(auto leg_id: orphaned_legs){
    output_tensor_rank = output_tensor->getNumLegs();
    auto dim_ext = output_tensor->getDimExtent(leg_id);
    auto dim_atr = output_tensor->getDimSpaceAttr(leg_id);
    auto delta_tensor_id = this->getMaxTensorId()+1; assert(delta_tensor_id > 0);
    auto appended = emplaceTensorConnPrefDirect(true,"d",false,delta_tensor_id,
                                                std::make_shared<Tensor>("_delta", //name will be changed
                                                 std::initializer_list<decltype(dim_ext)>{dim_ext,dim_ext},
                                                 std::initializer_list<decltype(dim_atr)>{dim_atr,dim_atr}),
                                                delta_tensor_id,
                                                std::vector<TensorLeg>{TensorLeg{0,leg_id},
                                                                       TensorLeg{0,output_tensor_rank+1}});
    assert(appended);
    output_tensor->resetLeg(leg_id,TensorLeg{delta_tensor_id,0});
    output_tensor->appendLeg(dim_atr,dim_ext,TensorLeg{delta_tensor_id,1});
   }
   this->updateConnections(0);
   if(deltas_appended != nullptr) *deltas_appended = true;
  }
 }
 //Delete the tensor from the network:
 tensor = nullptr;
 auto erased = eraseTensorConn(tensor_id); assert(erased);
 invalidateContractionSequence(); //invalidate previously cached tensor contraction sequence
 return true;
}


bool TensorNetwork::deleteKroneckerDeltas()
{
 auto predicate = [](const Tensor & tensor){
  bool is_delta_tensor = false;
  const auto & tensor_name = tensor.getName();
  if(tensor_name.length() >= 2){
   if(tensor_name[0] == '_' && tensor_name[1] == 'd') is_delta_tensor = true;
  }
  return is_delta_tensor;
 };
 const auto tensor_ids = getTensorIdsInNetwork(predicate);
 if(tensor_ids.size() > 0){
  for(const auto & id: tensor_ids){
   auto success = deleteTensor(id); assert(success);
  }
  return true;
 }
 return false;
}


bool TensorNetwork::mergeTensors(unsigned int left_id, unsigned int right_id, unsigned int result_id,
                                 std::string * contr_pattern)
{
 if(left_id == right_id || left_id == result_id || right_id == result_id){
  std::cout << "#ERROR(TensorNetwork::mergeTensors): Invalid arguments: Cannot be identical: " <<
   left_id << " " << right_id << " " << result_id << std::endl;
  return false;
 }
 if(left_id == 0 || right_id == 0 || result_id == 0){
  std::cout << "#ERROR(TensorNetwork::mergeTensors): Invalid arguments: Output tensor #0 cannot participate: " <<
   left_id << " " << right_id << " " << result_id << std::endl;
  return false;
 }
 if(finalized_ == 0){
  std::cout << "#ERROR(TensorNetwork::mergeTensors): Invalid request: " <<
   "Merging tensors in an unfinalized tensor network is forbidden!" << std::endl;
  return false;
 }
 //Get tensor info:
 auto * left_tensor = this->getTensorConn(left_id);
 assert(left_tensor != nullptr);
 auto left_tensor_rank = left_tensor->getNumLegs();
 auto left_tensor_conj = left_tensor->isComplexConjugated();
 const auto & left_legs = left_tensor->getTensorLegs();
 auto * right_tensor = this->getTensorConn(right_id);
 assert(right_tensor != nullptr);
 auto right_tensor_rank = right_tensor->getNumLegs();
 auto right_tensor_conj = right_tensor->isComplexConjugated();
 const auto & right_legs = right_tensor->getTensorLegs();
 //Count contracted and uncontracted legs:
 unsigned int num_contracted = 0;
 for(const auto & leg: left_legs){if(leg.getTensorId() == right_id) ++num_contracted;}
 unsigned int num_uncontracted = (left_legs.size() + right_legs.size()) - num_contracted*2;
 //Create the resulting legs and contraction pattern:
 std::vector<TensorLeg> result_legs(num_uncontracted,TensorLeg(0,0)); //placeholders for result-tensor legs
 std::vector<TensorLeg> pattern(left_legs.size()+right_legs.size(),TensorLeg(0,0)); //tensor contraction pattern (placeholder)
 unsigned int mode = 0;
 unsigned int res_mode = 0;
 for(const auto & leg: left_legs){
  if(leg.getTensorId() == right_id){ //contracted leg
   pattern[mode++] = TensorLeg(2,leg.getDimensionId());
  }else{ //uncontracted leg
   pattern[mode++] = TensorLeg(0,res_mode);
   result_legs[res_mode++] = leg;
  }
 }
 for(const auto & leg: right_legs){
  if(leg.getTensorId() == left_id){ //contracted leg
   pattern[mode++] = TensorLeg(1,leg.getDimensionId());
  }else{ //uncontracted leg
   pattern[mode++] = TensorLeg(0,res_mode);
   result_legs[res_mode++] = leg;
  }
 }
 assert(res_mode == num_uncontracted);
 //Generate symbolic contraction pattern if needed:
 if(contr_pattern != nullptr){
  auto generated = generate_contraction_pattern(pattern,left_tensor_rank,right_tensor_rank,
                                                *contr_pattern,left_tensor_conj,right_tensor_conj);
  assert(generated);
 }
 //Append the tensor result:
 auto res = emplaceTensorConnDirect(true,true,
                                    result_id,
                                    std::make_shared<Tensor>(
                                     "_y" + std::to_string(result_id), //this is temporary name, will change to _xHASH
                                     *(left_tensor->getTensor()),
                                     *(right_tensor->getTensor()),
                                     pattern
                                    ),
                                    result_id,result_legs);
 if(!res){
  std::cout << "#ERROR(exatn::numerics::TensorNetwork::mergeTensors): Unable to append the tensor-result!" << std::endl;
  return false;
 }
 //Delete two original tensors:
 auto erased = eraseTensorConn(left_id); assert(erased);
 erased = eraseTensorConn(right_id); assert(erased);
 //Update connections:
 this->updateConnections(result_id);
 invalidateContractionSequence(); //invalidate previously cached tensor contraction sequence
 return true;
}


bool TensorNetwork::mergeTensors(const std::vector<unsigned int> & tensor_ids,
                                 unsigned int result_id)
{
 bool success = true;
 //Check tensor ids:
 std::unordered_set<unsigned int> ids;
 for(const auto tens_id: tensor_ids){
  assert(tens_id != 0);
  auto res = ids.emplace(tens_id);
  assert(res.second);
 }
 //Extract the tensor sub-network:
 TensorNetwork subnetwork("_SubNetwork",*this,tensor_ids);
 //Replace the sub-network with a single tensor:
 TensorConn repl_tens_conn(*(subnetwork.getTensorConn(0)));
 repl_tens_conn.replaceStoredTensor();
 const auto repl_tens_rank = repl_tens_conn.getRank();
 for(unsigned int i = 0; i < repl_tens_rank; ++i){
  const auto & repl_tens_leg = repl_tens_conn.getTensorLeg(i);
  const auto tens_id = repl_tens_leg.getTensorId();
  const auto dim_id = repl_tens_leg.getDimensionId();
  auto * orig_tens_conn = getTensorConn(tens_id); assert(orig_tens_conn);
  const auto other_tens_leg = orig_tens_conn->getTensorLeg(dim_id);
  repl_tens_conn.resetLeg(i,other_tens_leg);
 }
 success = emplaceTensorConn(result_id,repl_tens_conn);
 //Erase merged tensors:
 if(success){
  for(const auto tens_id: tensor_ids){
   success = eraseTensorConn(tens_id);
   if(!success) break;
  }
  //Update connections:
  if(success){
   updateConnections(result_id);
   invalidateContractionSequence(); //invalidate previously cached tensor contraction sequence
  }
 }
 return success;
}


bool TensorNetwork::splitTensor(unsigned int tensor_id,
                                unsigned int left_tensor_id,
                                const std::string & left_tensor_name,
                                unsigned int right_tensor_id,
                                const std::string & right_tensor_name,
                                const TensorShape & contracted_dims,
                                const std::vector<int> & right_dims)
{
 //Check arguments:
 if(tensor_id == 0){
  std::cout << "#ERROR(TensorNetwork::splitTensor): Invalid request: " <<
   "Splitting the output tensor of the tensor network is forbidden!" << std::endl;
  return false;
 }
 if(left_tensor_id == 0 || right_tensor_id == 0 || left_tensor_id == right_tensor_id){
  std::cout << "#ERROR(TensorNetwork::splitTensor): Invalid request: " <<
   "Split tensors must acquire unique positive ids!" << std::endl;
  return false;
 }
 if(finalized_ == 0){
  std::cout << "#ERROR(TensorNetwork::splitTensor): Invalid request: " <<
   "Splitting a tensor in an unfinalized tensor network is forbidden!" << std::endl;
  return false;
 }
 auto * tensor = this->getTensorConn(tensor_id);
 assert(tensor != nullptr);
 const auto tensor_rank = tensor->getNumLegs();
 if(right_dims.size() != tensor_rank){
  std::cout << "#ERROR(TensorNetwork::splitTensor): Invalid request: " <<
   "The vector of tensor dimension split assignment has wrong size!" << std::endl;
  return false;
 }
 //Create new tensors from the original tensor:
 unsigned int left_rank = 0;
 unsigned int right_rank = 0;
 for(const auto & ass: right_dims) (ass == 0) ? ++left_rank : ++right_rank;
 unsigned int num_contr_dims = contracted_dims.getRank();
 unsigned int left_full_rank = left_rank + num_contr_dims;
 unsigned int right_full_rank = right_rank + num_contr_dims;
 auto left_tensor = tensor->getTensor()->createSubtensor(left_tensor_name,right_dims,0);
 assert(left_tensor);
 auto right_tensor = tensor->getTensor()->createSubtensor(right_tensor_name,right_dims,1);
 assert(right_tensor);
 std::vector<TensorLeg> left_legs(left_rank,TensorLeg(0,0));
 std::vector<TensorLeg> right_legs(right_rank,TensorLeg(0,0));
 for(unsigned int l = 0, r = 0, i = 0; i < tensor_rank; ++i){
  (right_dims[i] == 0) ? left_legs[l++] = tensor->getTensorLeg(i):
                         right_legs[r++] = tensor->getTensorLeg(i);
 }
 //Remove the original tensor from the tensor network:
 auto erased = eraseTensorConn(tensor_id); assert(erased);
 //Append the new derived tensors to the tensor network:
 auto res = emplaceTensorConnDirect(true,
                                    left_tensor_id,
                                    left_tensor,left_tensor_id,left_legs);
 if(!res){
  std::cout << "#ERROR(TensorNetwork::splitTensor): Invalid request: " <<
   "A tensor with id " << left_tensor_id << " already exists in the tensor network!" << std::endl;
  return false;
 }
 res = emplaceTensorConnDirect(true,
                               right_tensor_id,
                               right_tensor,right_tensor_id,right_legs);
 if(!res){
  std::cout << "#ERROR(TensorNetwork::splitTensor): Invalid request: " <<
   "A tensor with id " << right_tensor_id << " already exists in the tensor network!" << std::endl;
  return false;
 }
 updateConnections(left_tensor_id);
 updateConnections(right_tensor_id);
 //Append new (contracted) dimensions to both the left and right derived tensors:
 auto * left = this->getTensorConn(left_tensor_id); assert(left);
 auto * right = this->getTensorConn(right_tensor_id); assert(right);
 for(unsigned int i = 0; i < num_contr_dims; ++i){
  auto dim_extent = contracted_dims.getDimExtent(i);
  left->appendLeg(dim_extent,TensorLeg(right_tensor_id,right_rank++));
  right->appendLeg(dim_extent,TensorLeg(left_tensor_id,left_rank++));
 }
 assert(left_rank == left_full_rank && right_rank == right_full_rank);
 invalidateContractionSequence(); //invalidate previously cached tensor contraction sequence
 return true;
}


bool TensorNetwork::substituteTensor(unsigned int tensor_id, std::shared_ptr<Tensor> tensor)
{
 assert(tensor);
 auto * old_tensor_conn = this->getTensorConn(tensor_id);
 if(old_tensor_conn == nullptr) return false;
 if(!(tensor->isCongruentTo(*(old_tensor_conn->getTensor())))) return false;
 old_tensor_conn->replaceStoredTensor(tensor);
 return true;
}


bool TensorNetwork::substituteTensor(const std::string & name, std::shared_ptr<Tensor> tensor)
{
 assert(name.length() > 0);
 bool success = true;
 for(auto & tens: tensors_){
  if(tens.second.getName() == name){
   success = substituteTensor(tens.first,tensor);
   if(!success) break;
  }
 }
 return success;
}


bool TensorNetwork::substituteTensor(std::shared_ptr<Tensor> original,
                                     std::shared_ptr<Tensor> tensor)
{
 if(!(original->isCongruentTo(*tensor))) return false;
 for(auto & tens: tensors_){
  auto net_tensor = tens.second.getTensor();
  if(net_tensor == original) tens.second.replaceStoredTensor(tensor);
 }
 return true;
}


bool TensorNetwork::substituteTensor(unsigned int tensor_id,
                                     const TensorNetwork & network)
{
 bool success = true;
 const auto * out_tens_conn = const_cast<TensorNetwork&>(network).getTensorConn(0);
 assert(out_tens_conn != nullptr);
 const auto * repl_tens_conn = getTensorConn(tensor_id);
 if(repl_tens_conn != nullptr){
  if(repl_tens_conn->getTensor()->isCongruentTo(*(out_tens_conn->getTensor()))){
   //Establish tensor id renumeration map:
   unsigned int n = getMaxTensorId();
   std::unordered_map<unsigned int, unsigned int> id_map; //old id --> new id
   for(auto tens_entry = network.cbegin(); tens_entry != network.cend(); ++tens_entry){
    auto res = id_map.emplace(std::make_pair(tens_entry->first,++n)); assert(res.second);
   }
   //Emplace sub-network into the tensor network:
   for(auto tens_entry = network.cbegin(); tens_entry != network.cend(); ++tens_entry){
    TensorConn tens_conn(tens_entry->second);
    const auto tens_conn_rank = tens_conn.getRank();
    for(unsigned int i = 0; i < tens_conn_rank; ++i){
     auto tens_conn_leg = tens_conn.getTensorLeg(i);
     const auto other_tens_id = tens_conn_leg.getTensorId();
     const auto other_tens_dim = tens_conn_leg.getDimensionId();
     if(other_tens_id == 0){ //boundary leg
      tens_conn_leg.resetTensorId(repl_tens_conn->getTensorLeg(other_tens_dim).getTensorId());
      tens_conn_leg.resetDimensionId(repl_tens_conn->getTensorLeg(other_tens_dim).getDimensionId());
     }else{ //internal leg
      tens_conn_leg.resetTensorId(id_map[other_tens_id]);
     }
     tens_conn.resetLeg(i,tens_conn_leg);
    }
    success = emplaceTensorConn(id_map[tens_entry->first],tens_conn);
    if(!success) return false;
   }
   success = eraseTensorConn(tensor_id);
   if(success){
    for(auto tens_entry = network.cbegin(); tens_entry != network.cend(); ++tens_entry){
     updateConnections(id_map[tens_entry->first]);
    }
   }
   invalidateContractionSequence(); //invalidate previously cached tensor contraction sequence
  }else{
   success = false;
  }
 }else{
  success = false;
 }
 return success;
}


bool TensorNetwork::substituteTensor(const std::string & name,
                                     const TensorNetwork & network)
{
 assert(name.length() > 0);

 auto matcher = [&name] (const Tensor & tensor) {
  bool matched = false;
  if(tensor.getName() == name) matched = true;
  return matched;
 };

 bool success = true;
 const auto ids = getTensorIdsInNetwork(matcher);
 for(const auto id: ids){
  success = substituteTensor(id,network);
  if(!success) break;
 }
 return success;
}


std::vector<unsigned int> TensorNetwork::getTensorIdsInNetwork(const std::string & name, bool conjugated) const
{
 assert(name.length() > 0);
 std::vector<unsigned int> ids;
 for(const auto & kv: tensors_){
  if(kv.second.getName() == name &&
     kv.second.isComplexConjugated() == conjugated) ids.emplace_back(kv.first);
 }
 return ids;
}


std::vector<unsigned int> TensorNetwork::getTensorIdsInNetwork(std::function<bool (const Tensor &)> predicate) const
{
 std::vector<unsigned int> tensor_ids;
 for(const auto & tensor: tensors_){
  if(tensor.first != 0){ //ignore the output tensor of the tensor network
   if(predicate(*(tensor.second.getTensor()))) tensor_ids.emplace_back(tensor.first);
  }
 }
 return tensor_ids;
}


bool TensorNetwork::conjugate()
{
 if(finalized_ == 0){
  std::cout << "#ERROR(TensorNetwork::conjugate): Invalid request: " <<
   "Unfinalized tensor network may not be conjugated!" << std::endl;
  return false;
 }
 for(auto iter = this->begin(); iter != this->end(); ++iter) (iter->second).conjugate();
 return true;
}


bool TensorNetwork::collapseIsometries(bool * deltas_appended)
{
 if(deltas_appended != nullptr) *deltas_appended = false;
 bool simplified = false;
 if(finalized_ == 0){
  std::cout << "#ERROR(TensorNetwork::collapseIsometries): Invalid request: " <<
   "Unfinalized tensor network may not be simplified!" << std::endl;
  return simplified;
 }
 auto * output_tensor = this->getTensorConn(0); assert(output_tensor != nullptr);
 auto another_collapse = true;
 while(another_collapse){
  another_collapse = false;
  auto iter = this->begin();
  while(!another_collapse && iter != this->end()){
   if(iter->first != 0){ //only input tensors can collapse
    auto & tensor = iter->second; //connected tensor pointed to by the iterator
    const auto tensor_id = tensor.getTensorId();
    const auto & tensor_name = tensor.getName();
    const auto & tensor_legs = tensor.getTensorLegs(); //legs of the connected tensor
    const auto & tensor_isometries = tensor.retrieveIsometries(); //isometries of the connected tensor
    for(const auto & iso_group: tensor_isometries){
     bool iso_match = !(iso_group.empty());
     unsigned int iso_matched_tensor_id = 0;
     for(const auto & dimsn: iso_group){
      const auto other_tensor_id = tensor_legs[dimsn].getTensorId();
      const auto other_tensor_dimsn = tensor_legs[dimsn].getDimensionId();
      if(other_tensor_dimsn == dimsn){
       auto & other_tensor = *(this->getTensorConn(other_tensor_id)); //other connected tensor
       if(other_tensor.getName() == tensor_name){
        if((tensor.isComplexConjugated() && !(other_tensor.isComplexConjugated())) ||
           (!(tensor.isComplexConjugated()) && other_tensor.isComplexConjugated())){
         if(iso_matched_tensor_id == 0) iso_matched_tensor_id = other_tensor_id;
         if(other_tensor_id != iso_matched_tensor_id){
          iso_match = false;
          break;
         }
        }else{
         iso_match = false;
         break;
        }
       }else{
        iso_match = false;
        break;
       }
      }else{
       iso_match = false;
       break;
      }
     }
     if(iso_match && iso_matched_tensor_id > 0){ // [Tensor---Isometric_group---Tensor+] detected
      auto & conj_tensor = *(this->getTensorConn(iso_matched_tensor_id)); //conjugate tensor
      if(tensor.isCongruentTo(conj_tensor)){
       const auto & conj_tensor_legs = conj_tensor.getTensorLegs();
       unsigned int num_iso_legs = 0;
       unsigned int num_open_legs = 0;
       unsigned int num_spectators = 0;
       for(unsigned int i = 0; i < tensor.getNumLegs(); ++i){
        if(tensor_legs[i].getTensorId() == iso_matched_tensor_id) num_iso_legs++;
        if(tensor_legs[i].getTensorId() == 0 || conj_tensor_legs[i].getTensorId() == 0) num_open_legs++;
        if(tensor_legs[i].getTensorId() == 0 && conj_tensor_legs[i].getTensorId() == 0) num_spectators++;
       }
       //std::cout << "#DEBUG(collapseIsometries): Tensor pair: " << tensor_id << " " << iso_matched_tensor_id
       //          << ": " << iso_group.size() << " " << num_iso_legs << " " << num_open_legs << " " << num_spectators << std::endl;
       if(num_iso_legs == iso_group.size()){ //a single isometric connection between tensors identified: Collapse
        if(num_iso_legs < tensor.getNumLegs()){ //there are other legs other than isometric legs
         if(num_open_legs > 0) this->resetOutputTensor(); //new open legs need to be appended to the output tensor
         for(unsigned int i = 0; i < tensor.getNumLegs(); ++i){
          auto first_tensor_id = tensor_legs[i].getTensorId();
          if(first_tensor_id != iso_matched_tensor_id){
           auto first_tensor_dimsn = tensor_legs[i].getDimensionId();
           auto second_tensor_id = conj_tensor_legs[i].getTensorId();
           assert(second_tensor_id != tensor_id);
           auto second_tensor_dimsn = conj_tensor_legs[i].getDimensionId();
           if(first_tensor_id == 0 && second_tensor_id == 0){ //insert an explicit Delta tensor for a spectator
            auto output_tensor_rank = output_tensor->getNumLegs();
            auto dim_ext0 = output_tensor->getDimExtent(first_tensor_dimsn);
            auto dim_atr0 = output_tensor->getDimSpaceAttr(first_tensor_dimsn);
            auto dim_ext1 = output_tensor->getDimExtent(second_tensor_dimsn);
            auto dim_atr1 = output_tensor->getDimSpaceAttr(second_tensor_dimsn);
            assert(dim_ext0 == dim_ext1);
            assert(dim_atr0 == dim_atr1);
            auto delta_tensor = std::make_shared<Tensor>("_delta", //name will be changed to _dHash
                                 std::initializer_list<decltype(dim_ext0)>{dim_ext0,dim_ext1},
                                 std::initializer_list<decltype(dim_atr0)>{dim_atr0,dim_atr1});
            delta_tensor->setElementType(tensor.getElementType());
            auto delta_tensor_id = this->getMaxTensorId()+1; assert(delta_tensor_id > 0);
            auto appended = emplaceTensorConnPrefDirect(true,"d",false,delta_tensor_id,
                                                        delta_tensor, delta_tensor_id,
                                                        std::vector<TensorLeg>{TensorLeg{0,first_tensor_dimsn},
                                                                               TensorLeg{0,second_tensor_dimsn}});
            assert(appended);
            output_tensor->resetLeg(first_tensor_dimsn,TensorLeg{delta_tensor_id,0});
            output_tensor->resetLeg(second_tensor_dimsn,TensorLeg{delta_tensor_id,1});
            if(deltas_appended != nullptr) *deltas_appended = true;
           }else{ //Delta tensor is absorbed into the adjacent input tensor
            auto leg = this->getTensorConn(first_tensor_id)->getTensorLeg(first_tensor_dimsn);
            leg.resetTensorId(second_tensor_id);
            leg.resetDimensionId(second_tensor_dimsn);
            this->getTensorConn(first_tensor_id)->resetLeg(first_tensor_dimsn,leg);
            leg = this->getTensorConn(second_tensor_id)->getTensorLeg(second_tensor_dimsn);
            leg.resetTensorId(first_tensor_id);
            leg.resetDimensionId(first_tensor_dimsn);
            this->getTensorConn(second_tensor_id)->resetLeg(second_tensor_dimsn,leg);
           }
          }
         }
        }else{ //no other legs, this isometry collapses to a unity scalar
         auto scalar_tensor = std::make_shared<Tensor>("_e1");
         scalar_tensor->setElementType(tensor.getElementType());
         auto scalar_tensor_id = this->getMaxTensorId()+1; assert(scalar_tensor_id > 0);
         auto appended = emplaceTensorConnDirect(true,scalar_tensor_id,
                                                 scalar_tensor,scalar_tensor_id,
                                                 std::vector<TensorLeg>{});
         assert(appended);
        }
        //std::cout << "#DEBUG(collapseIsometries): Type 1 collapse: " << tensor_id << " " << iso_matched_tensor_id << std::endl; //debug
        auto erased = eraseTensorConn(tensor_id); assert(erased);
        erased = eraseTensorConn(iso_matched_tensor_id); assert(erased);
        another_collapse = true;
        simplified = true;
       }else if(num_iso_legs > iso_group.size() && num_iso_legs == tensor.getNumLegs()){ //potential double isometric connection between tensors (delta trace)
        std::size_t aux_vol = 1;
        bool double_isometry = true;
        for(unsigned int i = 0; i < tensor.getNumLegs(); ++i){
         if(tensor.withIsometricDimension(i)){
          if(tensor_legs[i].getDimensionId() != i){
           double_isometry = false;
           break;
          }
         }else{
          aux_vol *= tensor.getDimExtent(i);
         }
        }
        if(double_isometry){ //contract the isometric tensor pair into the Delta tensor
         for(unsigned int i = 0; i < tensor.getNumLegs(); ++i){
          assert(tensor_legs[i].getDimensionId() == i);
          assert(conj_tensor_legs[i].getDimensionId() == i);
          if(std::find(iso_group.cbegin(),iso_group.cend(),i) == iso_group.cend()){ //trace auxiliary delta leg
           auto dim_ext = tensor.getDimExtent(i);
           auto dim_atr = tensor.getDimSpaceAttr(i);
           auto scalar_tensor = std::make_shared<Tensor>("_e"+std::to_string(dim_ext));
           scalar_tensor->setElementType(tensor.getElementType());
           auto scalar_tensor_id = this->getMaxTensorId()+1; assert(scalar_tensor_id > 0);
           auto appended = emplaceTensorConnDirect(true,scalar_tensor_id,
                                                   scalar_tensor,scalar_tensor_id,
                                                   std::vector<TensorLeg>{});
           /*auto delta_tensor_id = this->getMaxTensorId()+1; assert(delta_tensor_id > 0);
           auto appended = emplaceTensorConnPrefDirect(true,"d",false,delta_tensor_id,
                                                       std::make_shared<Tensor>("_delta", //name will be changed
                                                        std::initializer_list<decltype(dim_ext)>{dim_ext,dim_ext},
                                                        std::initializer_list<decltype(dim_atr)>{dim_atr,dim_atr}),
                                                       delta_tensor_id,
                                                       std::vector<TensorLeg>{TensorLeg{delta_tensor_id,1},
                                                                              TensorLeg{delta_tensor_id,0}});*/
           assert(appended);
           //if(deltas_appended != nullptr) *deltas_appended = true;
          }
         }
         //std::cout << "#DEBUG(collapseIsometries): Type 2 collapse: " << tensor_id << " " << iso_matched_tensor_id << std::endl; //debug
         auto erased = eraseTensorConn(tensor_id); assert(erased);
         erased = eraseTensorConn(iso_matched_tensor_id); assert(erased);
         another_collapse = true;
         simplified = true;
        }
       }
      }
     }
     if(another_collapse) break;
    }
   }
   if(!another_collapse) ++iter;
  }
 }
 if(simplified) invalidateContractionSequence(); //invalidate previously cached tensor contraction sequence
 return simplified;
}


bool TensorNetwork::decomposeTensors()
{
 if(finalized_ == 0){
  std::cout << "#ERROR(TensorNetwork::decomposeTensors): Invalid request: " <<
   "Unfinalized tensor network may not be decomposed!" << std::endl;
  return false;
 }
 bool splitting = true;
 while(splitting){
  splitting = false;
  for(auto iter = this->begin(); iter != this->end(); ++iter){
   const auto tensor_id = iter->first;
   const auto & tensor = *((iter->second).getTensor());
   const auto tensor_rank = tensor.getRank();
   if(tensor_rank > 3){
    const auto left_rank = tensor_rank / 2;
    const auto right_rank = tensor_rank - left_rank;
    const auto & extents = tensor.getDimExtents();
    auto cmp = [&extents](const int & i1, const int & i2) {return extents[i1] < extents[i2];};
    std::vector<int> dims(tensor_rank);
    for(int i = 0; i < tensor_rank; ++i) dims[i] = i;
    std::sort(dims.begin(),dims.end(),cmp);
    std::vector<int> right_dims(tensor_rank);
    for(int i = 0; i < left_rank; ++i) right_dims[dims[i]] = 0;
    for(int i = left_rank; i < tensor_rank; ++i) right_dims[dims[i]] = 1;
    DimExtent contr_dim = 1; for(int i = 0; i < left_rank; ++i) contr_dim *= extents[dims[i]];
    const auto left_tensor_id = this->getMaxTensorId() + 1;
    const auto right_tensor_id = this->getMaxTensorId() + 2;
    splitting = this->splitTensor(tensor_id,left_tensor_id,"_left",right_tensor_id,"_right",
                                  TensorShape(std::initializer_list<DimExtent>{contr_dim}),right_dims);
    assert(splitting);
    this->getTensor(left_tensor_id)->rename(); //automatic unique name
    this->getTensor(right_tensor_id)->rename(); //automatic unique name
    break;
   }
  }
 }
 return true;
}


bool TensorNetwork::resetBondAdaptivity(std::shared_ptr<BondAdaptivity> bond_adaptivity)
{
 if(finalized_ == 0){
  std::cout << "#ERROR(TensorNetwork::resetBondAdaptivity): Invalid request: " <<
   "Unfinalized tensor network cannot have bond adaptivity policy!" << std::endl;
  return false;
 }
 bond_adaptivity_ = bond_adaptivity;
 return true;
}


bool TensorNetwork::applyBondAdaptivityStep(bool invalidate)
{
 bool success = true, adapted = false;
 if(bond_adaptivity_){
  for(const auto & policy: bond_adaptivity_->bond_policy_){
   const auto tid1 = policy.bond.first.getTensorId();
   const auto lid1 = policy.bond.first.getDimensionId();
   const auto tid2 = policy.bond.second.getTensorId();
   const auto lid2 = policy.bond.second.getDimensionId();
   const auto * tensor1 = getTensorConn(tid1);
   const auto * tensor2 = getTensorConn(tid2);
   if(tensor1 == nullptr || tensor2 == nullptr){
    std::cout << "#ERROR(TensorNetwork::applyBondAdaptivityStep): Invalid policy: " <<
     "Bond adaptivity policy refers to non-existing tensors: " << tid1 << " " << tid2 << std::endl;
    return false;
   }
   if(tensor1->getTensorLeg(lid1).getTensorId() != tid2 ||
      tensor1->getTensorLeg(lid1).getDimensionId() != lid2){
    std::cout << "#ERROR(TensorNetwork::applyBondAdaptivityStep): Invalid policy: " <<
     "Bond adaptivity policy refers to a non-existing bond between two tensors: " <<
     tid1 << " " << tid2 << std::endl;
    return false;
   }
   const auto dim_ext = tensor1->getDimExtent(lid1);
   const auto new_dim_ext = policy.adapt(dim_ext);
   if(new_dim_ext != dim_ext){
    auto new_tensor1 = makeSharedTensor(*(tensor1->getTensor()));
    new_tensor1->replaceDimension(lid1,new_dim_ext);
    new_tensor1->rename();
    auto new_tensor2 = makeSharedTensor(*(tensor2->getTensor()));
    new_tensor2->replaceDimension(lid2,new_dim_ext);
    new_tensor2->rename();
    success = substituteTensor(tid1,new_tensor1);
    success = substituteTensor(tid2,new_tensor2);
    adapted = true;
   }
  }
  if(adapted){
   if(invalidate){
    invalidateContractionSequence();
   }else{
    invalidateTensorOperationList();
   }
  }
 }else{
  success = false;
 }
 return success;
}


bool TensorNetwork::partition(std::size_t num_parts,  //in: desired number of parts
                              double imbalance,       //in: tolerated partition weight imbalance
                              std::vector<std::pair<std::size_t,std::vector<std::size_t>>> & parts, //out: partitions
                              std::size_t * edge_cut, //out: achieved edge cut value
                              std::size_t * num_cross_edges) const //out: total number of cross edges
{
 MetisGraph graph(*this);
 bool success = graph.partitionGraph(num_parts,imbalance);
 if(success){
  parts.resize(num_parts);
  const std::vector<idx_t> * part_weights = nullptr;
  const std::vector<idx_t> * renumbering = nullptr;
  const auto & partitions = graph.getPartitions(edge_cut,num_cross_edges,&part_weights,&renumbering);
  for(std::size_t i = 0; i < num_parts; ++i) parts[i].first = (*part_weights)[i]; //partition weights
  if(renumbering == nullptr){
   for(std::size_t vertex = 0; vertex < partitions.size(); ++vertex){
    const auto & part_id = partitions[vertex]; assert(part_id < num_parts);
    parts[part_id].second.emplace_back(vertex); //vertex id
   }
  }else{
   for(std::size_t vertex = 0; vertex < partitions.size(); ++vertex){
    const auto & part_id = partitions[vertex]; assert(part_id < num_parts);
    parts[part_id].second.emplace_back((*renumbering)[vertex]); //original tensor id
   }
  }
 }
 return success;
}


void TensorNetwork::markOptimizableTensors(std::function<bool (const Tensor &)> predicate)
{
 for(auto iter = this->begin(); iter != this->end(); ++iter){
  auto & tensor_conn = iter->second;
  if(tensor_conn.getTensorId() != 0) //output tensor cannot be optimizable
   tensor_conn.resetOptimizability(predicate(*(tensor_conn.getTensor())));
 }
 return;
}


void TensorNetwork::markOptimizableAllTensors()
{
 return markOptimizableTensors([](const Tensor &){return true;});
}


void TensorNetwork::markOptimizableNoTensors()
{
 return markOptimizableTensors([](const Tensor &){return false;});
}


void TensorNetwork::markOptimizableTensor(unsigned int tensor_id, bool optimizable)
{
 auto * tensor_conn = getTensorConn(tensor_id);
 make_sure(tensor_conn != nullptr,
  "#ERROR(TensorNetwork::markOptimizableTensor): Tensor "+std::to_string(tensor_id)+" not found!");
 tensor_conn->resetOptimizability(optimizable);
 return;
}


double TensorNetwork::getContractionCost(unsigned int left_id, unsigned int right_id,
                                         double * total_volume, double * diff_volume,
                                         double * arithm_intensity, bool adjust_cost)
{
 double flops = -1.0; //error
 if(left_id != 0 && right_id != 0){
  if(left_id != right_id){
   const auto * left_tensor = this->getTensorConn(left_id);
   assert(left_tensor != nullptr);
   const auto * right_tensor = this->getTensorConn(right_id);
   assert(right_tensor != nullptr);
   flops = getTensorContractionCost(*left_tensor,*right_tensor,total_volume,diff_volume,arithm_intensity,adjust_cost);
  }else{
   std::cout << "#ERROR(TensorNetwork::getContractionCost): Invalid request: "
             << "Two tensors to be contracted are identical!" << std::endl;
  }
 }else{
  std::cout << "#ERROR(TensorNetwork::getContractionCost): Invalid request: "
            << "The output tensor of the tensor network (tensor 0) cannot be contracted!" << std::endl;
 }
 return flops;
}


std::list<std::shared_ptr<TensorOperation>> & TensorNetwork::getOperationList(const std::string & contr_seq_opt_name,
                                                                              bool universal_indices)
{
 const auto default_elem_type = getTensorElementType();
 if(operations_.empty()){
  //Determine the pseudo-optimal sequence of tensor contractions:
  max_intermediate_presence_volume_ = 0.0;
  max_intermediate_volume_ = 0.0;
  max_intermediate_rank_ = 0;
  double flops = determineContractionSequence();
  //Generate the list of operations (tensor contractions):
  std::size_t intermediates_vol = 0;
  auto & tensor_op_factory = *(TensorOpFactory::get());
  if(this->getNumTensors() > 1){ //two or more input tensors: One or more contractions
   TensorNetwork net(*this);
   std::list<unsigned int> intermediates;
   unsigned int num_contractions = contraction_seq_.size();
   for(auto contr = contraction_seq_.cbegin(); contr != contraction_seq_.cend(); ++contr){
    //std::cout << "#DEBUG(TensorNetwork::getOperationList): Contracting " << contr->left_id << " * " << contr->right_id
    //          << " -> " << contr->result_id << std::endl; //debug
    bool conj1, conj2;
    auto tensor1 = net.getTensor(contr->left_id,&conj1);
    auto tensor2 = net.getTensor(contr->right_id,&conj2);
    std::string contr_pattern;
    if(num_contractions > 1){ //intermediate contraction
     auto merged = net.mergeTensors(contr->left_id,contr->right_id,contr->result_id,&contr_pattern); //append intermediate _xHASH
     assert(merged);
    }else{ //last contraction
     assert(contr->result_id == 0); //last tensor contraction accumulates into the output tensor of the tensor network
     const auto * tensor1_legs = net.getTensorConnections(contr->left_id);
     assert(tensor1_legs != nullptr);
     const auto * tensor2_legs = net.getTensorConnections(contr->right_id);
     assert(tensor2_legs != nullptr);
     std::vector<TensorLeg> pattern(*tensor1_legs);
     pattern.insert(pattern.end(),tensor2_legs->begin(),tensor2_legs->end());
     auto generated = generate_contraction_pattern(pattern,tensor1_legs->size(),tensor2_legs->size(),
                                                   contr_pattern,conj1,conj2);
     assert(generated);
    }
    auto tensor0 = net.getTensor(contr->result_id);
    max_intermediate_volume_ = std::max(max_intermediate_volume_,static_cast<double>(tensor0->getVolume()));
    max_intermediate_rank_ = std::max(max_intermediate_rank_,tensor0->getRank());
    if(contr->result_id != 0){ //intermediate tensors need to be created/destroyed
     intermediates_vol += tensor0->getVolume();
     max_intermediate_presence_volume_ = std::max(max_intermediate_presence_volume_,static_cast<double>(intermediates_vol));
     auto op_create = tensor_op_factory.createTensorOpShared(TensorOpCode::CREATE); //create intermediate
     op_create->setTensorOperand(tensor0);
     if(tensor0->getElementType() != TensorElementType::VOID){
      std::dynamic_pointer_cast<TensorOpCreate>(op_create)->resetTensorElementType(tensor0->getElementType());
     }else{
      std::dynamic_pointer_cast<TensorOpCreate>(op_create)->resetTensorElementType(default_elem_type);
     }
     operations_.emplace_back(op_create);
     intermediates.emplace_back(contr->result_id);
     if(ACCUMULATIVE_CONTRACTIONS){
      std::shared_ptr<TensorOperation> op_init(std::move(tensor_op_factory.createTensorOp(TensorOpCode::TRANSFORM))); //init intermediate to zero
      op_init->setTensorOperand(tensor0);
      std::dynamic_pointer_cast<TensorOpTransform>(op_init)->
           resetFunctor(std::shared_ptr<talsh::TensorFunctor<Identifiable>>(new FunctorInitVal(0.0)));
      operations_.emplace_back(op_init);
     }
    }else{ //make sure the output tensor has its type set
     if(tensor0->getElementType() == TensorElementType::VOID) tensor0->setElementType(default_elem_type);
    }
    auto op = tensor_op_factory.createTensorOpShared(TensorOpCode::CONTRACT);
    op->setTensorOperand(tensor0);
    op->setTensorOperand(tensor1,conj1);
    op->setTensorOperand(tensor2,conj2);
    op->setIndexPattern(contr_pattern);
    if(!ACCUMULATIVE_CONTRACTIONS && contr->result_id != 0) //no accumulation into intermediate tensors
     std::dynamic_pointer_cast<TensorOpContract>(op)->resetAccumulative(false);
    assert(op->isSet());
    operations_.emplace_back(op);
    auto left_intermediate = std::find(intermediates.begin(),intermediates.end(),contr->left_id);
    if(left_intermediate != intermediates.end()){
     intermediates_vol -= tensor1->getVolume();
     auto op_destroy = tensor_op_factory.createTensorOp(TensorOpCode::DESTROY); //destroy intermediate
     op_destroy->setTensorOperand(tensor1);
     operations_.emplace_back(std::shared_ptr<TensorOperation>(std::move(op_destroy)));
     intermediates.erase(left_intermediate);
    }
    auto right_intermediate = std::find(intermediates.begin(),intermediates.end(),contr->right_id);
    if(right_intermediate != intermediates.end()){
     intermediates_vol -= tensor2->getVolume();
     auto op_destroy = tensor_op_factory.createTensorOp(TensorOpCode::DESTROY); //destroy intermediate
     op_destroy->setTensorOperand(tensor2);
     operations_.emplace_back(std::shared_ptr<TensorOperation>(std::move(op_destroy)));
     intermediates.erase(right_intermediate);
    }
    --num_contractions;
   }
   assert(intermediates.empty());
   assert(intermediates_vol == 0);
  }else{ //one input tensor: Single addition
   std::shared_ptr<Tensor> tensor0(nullptr);
   std::shared_ptr<Tensor> tensor1(nullptr);
   bool conj1;
   unsigned int left_tensor_id = 0;
   for(auto iter = this->begin(); iter != this->end(); ++iter){
    if(iter->first == 0){
     tensor0 = this->getTensor(iter->first);
    }else{
     tensor1 = this->getTensor(iter->first,&conj1);
     left_tensor_id = iter->first;
    }
   }
   if(tensor0->getElementType() == TensorElementType::VOID) tensor0->setElementType(default_elem_type);
   auto op = tensor_op_factory.createTensorOp(TensorOpCode::ADD);
   op->setTensorOperand(tensor0);
   op->setTensorOperand(tensor1,conj1);
   const auto * tensor1_legs = this->getTensorConnections(left_tensor_id);
   assert(tensor1_legs != nullptr);
   std::string contr_pattern;
   auto generated = generate_addition_pattern(*tensor1_legs,contr_pattern,conj1);
   assert(generated);
   op->setIndexPattern(contr_pattern);
   assert(op->isSet());
   operations_.emplace_back(std::shared_ptr<TensorOperation>(std::move(op)));
  }
  //std::cout << "#DEBUG(exatn::numerics::TensorNetwork::getOperationList): Flop count = " << flops
  //          << "; Max intermediate presence volume = " << max_intermediate_presence_volume_
  //          << "; Max intermediate volume = " << max_intermediate_volume_
  //          << "; Max intermediate rank = " << max_intermediate_rank_ << std::endl; //debug
 }
 if(universal_indices) establishUniversalIndexNumeration();
 return operations_;
}


void TensorNetwork::splitIndices(std::size_t max_intermediate_volume)
{
 assert(!operations_.empty());

 std::unordered_map<std::string,   //index name
                    double         //cumulative volume of all intermediates carrying this index
                   > index_volume; //index name --> cumulative index volume

 std::unordered_map<std::string,            //index label
                    std::pair<unsigned int, //global index id
                              IndexSplit>   //splitting info (segment composition)
                   > splitted; //info on splitted indices

 std::vector<std::pair<unsigned int, //dimension position in the tensor
                       std::size_t>  //number of segments to split into
            > dims; //for each tensor dimension

 std::vector<std::pair<unsigned int, //global id of the split index
                       unsigned int> //dimension position in the tensor
            > split_dims; //for each tensor dimension split

 std::vector<std::string> tens_operands; //extracted tensor operands
 std::vector<IndexLabel> indices; //indices extracted from a tensor
 std::string tens_name; //extracted tensor name
 bool conjugated = false;

 //Establish universal index numeration:
 split_tensors_.clear();
 split_indices_.clear();
 establishUniversalIndexNumeration();

 //Compute cumulative index volumes over all tensor operations:
 for(auto op_iter = operations_.cbegin(); op_iter != operations_.cend(); ++op_iter){
  const auto & op = *(*op_iter); //tensor operation
  const auto num_operands = op.getNumOperands();
  const auto & pattern = op.getIndexPattern();
  if(!pattern.empty()){ //tensor operation with two or more tensor operands (has a symbolic index pattern)
   //Extract symbolic tensor operands from the current tensor operation:
   tens_operands.clear();
   bool success = parse_tensor_network(pattern,tens_operands);
   if(success){
    assert(tens_operands.size() == num_operands);
    tens_name.clear(); indices.clear();
    success = parse_tensor(tens_operands[0],tens_name,indices,conjugated); //`Assumes a single output tensor operand (#0)
    if(success){
     assert(!conjugated); //output tensor operands never appear conjugated
     auto intermediate_p = op.getTensorOperand(0);
     if(intermediate_p == nullptr){ //trap
      std::cout << "#ERROR(exatn::TensorNetwork::splitIndices): Tensor operation is missing operand #0:" << std::endl;
      op.printIt();
      assert(false);
     }
     const auto & intermediate = *intermediate_p;
     const auto & intermediate_name = intermediate.getName();
     assert(intermediate_name == tens_name); //tensor must enter the symbolic index pattern under the same name
     assert(indices.size() == intermediate.getRank());
     double intermediate_volume = 1.0;
     for(unsigned int i = 0; i < indices.size(); ++i){
      intermediate_volume *= static_cast<double>(intermediate.getDimExtent(i)); //full dimension extent
     }
     for(unsigned int i = 0; i < indices.size(); ++i){
      auto res = index_volume.emplace(std::make_pair(indices[i].label,intermediate_volume));
      if(!(res.second)) res.first->second += intermediate_volume;
     }
    }else{
     std::cout << "#ERROR(exatn::numerics::TensorNetwork::splitIndices): "
               << "Unable to parse the output tensor operand: " << tens_operands[0] << std::endl;
     assert(false);
    }
   }else{
    std::cout << "#ERROR(exatn::numerics::TensorNetwork::splitIndices): "
              << "Unable to parse the tensor operation index pattern: " << pattern << std::endl;
    assert(false);
   }
  }
 }

 //Traverse tensor operations in reverse order and split intermediate tensor indices:
 unsigned int num_split_indices = 0; //total number of indices split
 for(auto op_iter = operations_.rbegin(); op_iter != operations_.rend(); ++op_iter){
  const auto & op = *(*op_iter); //tensor operation
  const auto op_hash = op.getTensorOpHash();
  const auto num_operands = op.getNumOperands();
  const auto num_operands_out = op.getNumOperandsOut();
  const auto & pattern = op.getIndexPattern();
  //Analyze the tensor operation with a symbolic index pattern:
  if(!pattern.empty()){ //tensor operation with two or more tensor operands (has a symbolic index pattern)
   assert(num_operands > 1 && num_operands_out == 1); //`Expecting only a single output tensor operand here (no SVDs, etc)
   //Extract symbolic tensor operands from the current tensor operation:
   tens_operands.clear();
   bool success = parse_tensor_network(pattern,tens_operands);
   if(success){
    assert(tens_operands.size() == num_operands);
    //Inspect the output tensor operand (intermediate tensor) and split its dimensions if needed:
    tens_name.clear(); indices.clear();
    success = parse_tensor(tens_operands[0],tens_name,indices,conjugated); //`Assumes a single output tensor operand (#0)
    if(success){
     assert(!conjugated); //output tensor operands never appear conjugated
     const auto & intermediate = *(op.getTensorOperand(0));
     const auto & intermediate_name = intermediate.getName();
     assert(intermediate_name == tens_name); //tensor must enter the symbolic index pattern under the same name
     assert(indices.size() == intermediate.getRank());
     /*if(!isIntermediateTensorName(tens_name)){ //output tensor operands must be intermediate tensors
      std::cout << "#ERROR(exatn::numerics::TensorNetwork::splitIndices): "
                << "The output tensor operand name does not name an intermediate tensor: " << tens_name << std::endl;
      assert(false);
     }*/
     //Compute the volume of the intermediate tensor and find its full dimensions:
     dims.clear();
     std::size_t intermediate_volume = 1;
     for(unsigned int i = 0; i < indices.size(); ++i){
      auto index_iter = splitted.find(indices[i].label);
      if(index_iter != splitted.end()){
       intermediate_volume *= index_iter->second.second[0].second; //segment extent (already split index)
      }else{
       intermediate_volume *= intermediate.getDimExtent(i); //full dimension extent
       dims.emplace_back(std::pair<unsigned int,std::size_t>{i,1});
      }
     }
     //Split the found full dimensions of the intermediate tensor:
     if(max_intermediate_volume > 0 && intermediate_volume > max_intermediate_volume){
      assert(dims.size() > 0); //at least one full dimension is expected
      //Prioritize full indices by their cumulative volume:
      std::stable_sort(dims.begin(),dims.end(),[&index_volume,&indices](const auto & d1, const auto & d2){
                                                return index_volume[indices[d1.first].label]
                                                     < index_volume[indices[d2.first].label];
                                               });
      //Reduce the volume of the intermediate tensor by increasing the number of segments per tensor dimensions:
      int i = dims.size() - 1; //split dimensions from the right (because of column-wise tensor storage)
      while(intermediate_volume > max_intermediate_volume){
       if((dims[i].second)*2 <= intermediate.getDimExtent(dims[i].first)){
        dims[i].second <<= 1; intermediate_volume >>= 1; //split tensor dimension in half
       }
       if(--i < 0) i = dims.size() - 1;
      }
      //Split full tensor dimensions into segments:
      for(const auto & dim: dims){
       const auto & num_dim_segs = dim.second;
       if(num_dim_segs > 1){ //number of segments
        const auto & dim_pos = dim.first;
        IndexSplit split_info = splitDimension(intermediate.getDimSpaceAttr(dim_pos),
                                               intermediate.getDimExtent(dim_pos),
                                               num_dim_segs);
        auto saved = splitted.emplace(std::make_pair(indices[dim_pos].label,
                                                     std::make_pair(num_split_indices,split_info)));
        assert(saved.second);
        split_indices_.emplace_back(std::make_pair(indices[dim_pos].label,split_info));
        num_split_indices++;
       }
      }
     }
    }else{
     std::cout << "#ERROR(exatn::numerics::TensorNetwork::splitIndices): "
               << "Unable to parse the output tensor operand: " << tens_operands[0] << std::endl;
     assert(false);
    }
   }else{
    std::cout << "#ERROR(exatn::numerics::TensorNetwork::splitIndices): "
              << "Unable to parse the tensor operation index pattern: " << pattern << std::endl;
    assert(false);
   }
  }
 }
 assert(split_indices_.size() == num_split_indices);

 //Traverse tensor operations in reverse order and mark index splitting in each affected tensor:
 for(auto op_iter = operations_.rbegin(); op_iter != operations_.rend(); ++op_iter){
  const auto & op = *(*op_iter); //tensor operation
  const auto op_hash = op.getTensorOpHash();
  const auto num_operands = op.getNumOperands();
  const auto num_operands_out = op.getNumOperandsOut();
  const auto & pattern = op.getIndexPattern();
  //Analyze the tensor operation with a symbolic index pattern:
  if(!pattern.empty()){
   //Extract symbolic tensor operands from the current tensor operation:
   tens_operands.clear();
   bool success = parse_tensor_network(pattern,tens_operands);
   if(success){
    assert(tens_operands.size() == num_operands);
    //Inspect all tensor operands and mark their splitted dimensions:
    for(unsigned int op_num = 0; op_num < num_operands; ++op_num){
     const auto & tensor = *(op.getTensorOperand(op_num));
     const auto tensor_hash = tensor.getTensorHash();
     const auto & tensor_name = tensor.getName();
     //Inspect indices of the tensor operand for having split dimensions:
     tens_name.clear(); indices.clear();
     success = parse_tensor(tens_operands[op_num],tens_name,indices,conjugated);
     if(success){
      assert(tens_name == tensor_name); //tensor must enter the symbolic index pattern under the same name
      split_dims.clear();
      for(unsigned int i = 0; i < indices.size(); ++i){
       auto index_iter = splitted.find(indices[i].label);
       if(index_iter != splitted.end()){
        split_dims.emplace_back(std::make_pair(index_iter->second.first,i));
       }
      }
      //Save the inferred dimension splitting info for the tensor operand:
      if(split_dims.size() > 0){
       //std::cout << "#DEBUG(exatn::numerics::TensorNetwork::splitIndices): Splitting tensor " << tens_operands[op_num] << " @ "; //debug
       //for(const auto & ind: split_dims) std::cout << " " << ind.second; std::cout << std::endl; //debug
       if(op_num == 0){ //output tensor operand: pure intermediate or output tensor `Assumes a single output tensor operand (#0)
        //Intermediate tensors (including the tensor network output) are identified by the tensor hash:
        const auto key = std::make_pair(static_cast<TensorHashType>(0),tensor_hash);
        auto saved = split_tensors_.emplace(std::make_pair(key,split_dims));
        assert(saved.second);
       }else{
        if(!isIntermediateTensorName(tensor_name)){ // input tensor operand
         //Input tensors are identified by the tensor operation hash and their position in it:
         const auto key = std::make_pair(op_hash,static_cast<TensorHashType>(op_num));
         auto saved = split_tensors_.emplace(std::make_pair(key,split_dims));
         assert(saved.second);
        }
       }
      }
     }else{
      std::cout << "#ERROR(exatn::numerics::TensorNetwork::splitIndices): "
                << "Unable to parse a tensor operand: " << tens_operands[op_num] << std::endl;
      assert(false);
     }
    }
   }else{
    std::cout << "#ERROR(exatn::numerics::TensorNetwork::splitIndices): "
              << "Unable to parse the tensor operation index pattern: " << pattern << std::endl;
    assert(false);
   }
  }
 }
 return;
}


unsigned int TensorNetwork::getNumSplitIndices() const
{
 return static_cast<unsigned int>(split_indices_.size());
}


const std::pair<std::string,IndexSplit> &
 TensorNetwork::getSplitIndexInfo(unsigned int global_index_id) const
{
 assert(global_index_id < split_indices_.size());
 return split_indices_[global_index_id];
}


const std::vector<std::pair<unsigned int, unsigned int>> *
 TensorNetwork::getSplitTensorInfo(const std::pair<TensorHashType,TensorHashType> & key) const
{
 auto iter = split_tensors_.find(key);
 if(iter != split_tensors_.end()) return &(iter->second);
 return nullptr;
}


void TensorNetwork::printSplitIndexInfo(bool with_affected_tensors) const
{
 std::cout << "#INFO(TensorNetwork::printSplitIndexInfo):\n";
 for(unsigned int i = 0; i < split_indices_.size(); ++i){
  std::cout << i << ": " << split_indices_[i].first << ": Number of segments = "
            << split_indices_[i].second.size() << ":";
  for(const auto & seg: split_indices_[i].second)
   std::cout << "{" << seg.first << ":" << seg.second << "}";
  std::cout << std::endl;
 }
 if(with_affected_tensors && split_indices_.size() > 0){
  std::cout << "Affected tensors in tensor operations:\n";
  for(const auto & op: operations_){
   bool op_affected = false;
   const auto num_operands = op->getNumOperands();
   for(unsigned int i = 0; i < num_operands; ++i){
    const auto & tens = *(op->getTensorOperand(i));
    auto iter = split_tensors_.cend();
    if(tensorNameIsIntermediate(tens) || (i == 0)){ //intermediate tensor (includes output tensor of the tensor network)
     const auto key = std::pair<TensorHashType,TensorHashType>{0,tens.getTensorHash()};
     iter = split_tensors_.find(key);
    }else{ //input tensor
     const auto key = std::pair<TensorHashType,TensorHashType>{op->getTensorOpHash(),i};
     iter = split_tensors_.find(key);
    }
    if(iter != split_tensors_.cend()){
     std::cout << "Tensor "; tens.printIt(); std::cout << ":";
     for(const auto & ind: iter->second)
      std::cout << " " << split_indices_[ind.first].first << "@" << ind.second;
     std::cout << std::endl;
     op_affected = true;
    }
   }
   if(op_affected){
    std::cout << "in tensor operation:\n";
    op->printIt();
   }
  }
 }
 std::cout << "#END INFO\n";
 return;
}


void TensorNetwork::printSplitIndexInfo(std::ofstream & output_file, bool with_affected_tensors) const
{
 output_file << "#INFO(TensorNetwork::printSplitIndexInfo):\n";
 for(unsigned int i = 0; i < split_indices_.size(); ++i){
  output_file << i << ": " << split_indices_[i].first << ": Number of segments = "
              << split_indices_[i].second.size() << ":";
  for(const auto & seg: split_indices_[i].second)
   output_file << "{" << seg.first << ":" << seg.second << "}";
  output_file << std::endl;
 }
 if(with_affected_tensors && split_indices_.size() > 0){
  output_file << "Affected tensors in tensor operations:\n";
  for(const auto & op: operations_){
   bool op_affected = false;
   const auto num_operands = op->getNumOperands();
   for(unsigned int i = 0; i < num_operands; ++i){
    const auto & tens = *(op->getTensorOperand(i));
    auto iter = split_tensors_.cend();
    if(tensorNameIsIntermediate(tens) || (i == 0)){ //intermediate tensor (includes output tensor of the tensor network)
     const auto key = std::pair<TensorHashType,TensorHashType>{0,tens.getTensorHash()};
     iter = split_tensors_.find(key);
    }else{ //input tensor
     const auto key = std::pair<TensorHashType,TensorHashType>{op->getTensorOpHash(),i};
     iter = split_tensors_.find(key);
    }
    if(iter != split_tensors_.cend()){
     output_file << "Tensor "; tens.printItFile(output_file); output_file << ":";
     for(const auto & ind: iter->second)
      output_file << " " << split_indices_[ind.first].first << "@" << ind.second;
     output_file << std::endl;
     op_affected = true;
    }
   }
   if(op_affected){
    output_file << "in tensor operation:\n";
    op->printItFile(output_file);
   }
  }
 }
 output_file << "#END INFO\n";
 return;
}


void TensorNetwork::printContractionSequence() const
{
 std::cout << "TensorNetwork " << name_ << ": Contraction sequence:" << std::endl;
 return exatn::numerics::printContractionSequence(contraction_seq_);
}


void TensorNetwork::printContractionSequence(std::ofstream & output_file) const
{
 output_file << "TensorNetwork " << name_ << ": Contraction sequence:" << std::endl;
 return exatn::numerics::printContractionSequence(output_file,contraction_seq_);
}


void TensorNetwork::printOperationList() const
{
 std::cout << "TensorNetwork " << name_ << ": Tensor operation list:" << std::endl;
 for(auto & op: operations_) op->printIt();
 return;
}


double TensorNetwork::getFMAFlops() const
{
 return contraction_seq_flops_;
}


double TensorNetwork::getMaxIntermediatePresenceVolume() const
{
 return max_intermediate_presence_volume_;
}


double TensorNetwork::getMaxIntermediateVolume(unsigned int * intermediate_rank) const
{
 if(intermediate_rank != nullptr) *intermediate_rank = max_intermediate_rank_;
 return max_intermediate_volume_;
}


bool TensorNetwork::printTensorNetwork(std::string & network)
{
 network.clear();
 if(!operations_.empty()){
  establishUniversalIndexNumeration();
  for(auto op_iter = operations_.cbegin(); op_iter != operations_.cend(); ++op_iter){
   const auto & op = *(*op_iter);
   const auto & pattern = op.getIndexPattern();
   if(!pattern.empty()) network += (pattern + "\n");
  }
  return true;
 }
 return false;
}

} //namespace numerics


numerics::TensorHashType getTensorNetworkHash(std::shared_ptr<numerics::TensorNetwork> network)
{
 return reinterpret_cast<numerics::TensorHashType>((void*)(network.get()));
}

} //namespace exatn
