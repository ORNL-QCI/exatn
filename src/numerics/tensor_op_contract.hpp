/** ExaTN::Numerics: Tensor operation: Contracts two tensors and accumulates the result into another tensor
REVISION: 2021/08/18

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Contracts two tensors and accumulates the result into another tensor
     inside the processing backend:
     Operand 0 += Operand 1 * Operand 2 * prefactor
**/

#ifndef EXATN_NUMERICS_TENSOR_OP_CONTRACT_HPP_
#define EXATN_NUMERICS_TENSOR_OP_CONTRACT_HPP_

#include "tensor_basic.hpp"
#include "tensor_operation.hpp"

#include <map>

namespace exatn{

namespace numerics{

class TensorOpContract: public TensorOperation{
public:

 TensorOpContract();

 TensorOpContract(const TensorOpContract &) = default;
 TensorOpContract & operator=(const TensorOpContract &) = default;
 TensorOpContract(TensorOpContract &&) noexcept = default;
 TensorOpContract & operator=(TensorOpContract &&) noexcept = default;
 virtual ~TensorOpContract() = default;

 virtual std::unique_ptr<TensorOperation> clone() const override{
  return std::unique_ptr<TensorOperation>(new TensorOpContract(*this));
 }

 /** Returns TRUE iff the tensor operation is fully set. **/
 virtual bool isSet() const override;

 /** Accepts tensor node executor which will execute this tensor operation. **/
 virtual int accept(runtime::TensorNodeExecutor & node_executor,
                    runtime::TensorOpExecHandle * exec_handle) override;

 /** Decomposes a composite tensor operation into simple ones.
     Returns the total number of generated simple operations. **/
 virtual std::size_t decompose(const TensorMapper & tensor_mapper) override;

 /** Returns the flop estimate for the tensor operation. **/
 virtual double getFlopEstimate() const override;

 /** Create a new polymorphic instance of this subclass. **/
 static std::unique_ptr<TensorOperation> createNew();

 /** Resets the accumulative attribute. **/
 void resetAccumulative(bool accum);

 /** Queries the accumulative attribute. **/
 inline bool isAccumulative() const {
  return accumulative_;
 }

 /** Replaces none, some, or all tensor operands with new temporary
     tensors with optimized distributed storage configuration
     (only for composite tensor contractions). **/
 void introduceOptTemporaries(unsigned int num_processes,                        //in: number of processes
                              std::size_t mem_per_process,                       //in: memory limit per process
                              const std::vector<PosIndexLabel> & left_indices,   //in: extracted left indices
                              const std::vector<PosIndexLabel> & right_indices,  //in: extracted right indices
                              const std::vector<PosIndexLabel> & contr_indices,  //in: extracted cotracted indices
                              const std::vector<PosIndexLabel> & hyper_indices); //in: extracted hyper indices

 void introduceOptTemporaries(unsigned int num_processes,   //in: number of processes
                              std::size_t mem_per_process); //in: memory limit per process

protected:

 //Index attribute:
 struct IndexAttr{
  IndexKind index_kind; //index kind: {LEFT,RIGHT,CONTR,HYPER}
  DimExtent extent;     //index dimension extent
  unsigned int depth;   //index splitting depth (number of bisections)
 };

 //Information on all indices of a tensor contraction:
 struct IndexInfo{
  const TensorOperation * host_tensor_operation;
  std::vector<std::pair<std::string,IndexAttr>> left_indices;
  std::vector<std::pair<std::string,IndexAttr>> right_indices;
  std::vector<std::pair<std::string,IndexAttr>> contr_indices;
  std::vector<std::pair<std::string,IndexAttr>> hyper_indices;
  std::map<std::string,const IndexAttr*> index_attr;

  IndexInfo(const TensorOpContract * tensor_contraction):
   host_tensor_operation(tensor_contraction)
  {assert(host_tensor_operation);}

  //Appends a new index:
  bool appendIndex(const PosIndexLabel & label){
   const auto ind_kind = indexKind(label);
   const auto & ind_name = label.index_label.label;
   DimExtent ind_extent = 0;
   unsigned int ind_depth = 0;
   for(int i = 0; i < 3; ++i){
    const auto ind_pos = label.arg_pos[i];
    if(ind_pos >= 0){
     auto tens_operand = host_tensor_operation->getTensorOperand(i);
     const auto dim_ext = tens_operand->getDimExtent(ind_pos);
     if(ind_extent == 0) ind_extent = dim_ext;
     assert(dim_ext == ind_extent);
     if(tens_operand->isComposite()){
      const auto dim_depth = castTensorComposite(tens_operand)->getDimDepth(ind_pos);
      ind_depth = std::max(ind_depth,dim_depth);
     }
    }
   }
   assert(ind_extent > 0);
   const IndexAttr * attr_ptr {nullptr};
   switch(ind_kind){
    case IndexKind::LEFT:
     left_indices.emplace_back(std::make_pair(ind_name,IndexAttr{ind_kind,ind_extent,ind_depth}));
     attr_ptr = &(left_indices.back().second);
     break;
    case IndexKind::RIGHT:
     right_indices.emplace_back(std::make_pair(ind_name,IndexAttr{ind_kind,ind_extent,ind_depth}));
     attr_ptr = &(right_indices.back().second);
     break;
    case IndexKind::CONTR:
     contr_indices.emplace_back(std::make_pair(ind_name,IndexAttr{ind_kind,ind_extent,ind_depth}));
     attr_ptr = &(contr_indices.back().second);
     break;
    case IndexKind::HYPER:
     hyper_indices.emplace_back(std::make_pair(ind_name,IndexAttr{ind_kind,ind_extent,ind_depth}));
     attr_ptr = &(hyper_indices.back().second);
     break;
    default:
     std::cout << "#ERROR(tensor_op_contract:IndexInfo:appendIndex): Invalid index kind: "
               << static_cast<int>(ind_kind) << std::endl << std::flush;
     assert(false);
   }
   auto res = index_attr.emplace(std::make_pair(ind_name,attr_ptr));
   return res.second;
  }

  //Retrieves a previously stored index:
  const IndexAttr * getIndex(const std::string & index_name){
   auto iter = index_attr.find(index_name);
   if(iter == index_attr.end()) return nullptr;
   return iter->second;
  }

  //Computes the combined dimension extent of a given index group (LEFT,RIGHT,CONTR,HYPER):
  DimExtent totalExtent(IndexKind index_group) const{
   DimExtent total_extent = 1;
   switch(index_group){
    case IndexKind::LEFT:
     for(const auto & ind: left_indices) total_extent *= ind.second.extent;
     break;
    case IndexKind::RIGHT:
     for(const auto & ind: right_indices) total_extent *= ind.second.extent;
     break;
    case IndexKind::CONTR:
     for(const auto & ind: contr_indices) total_extent *= ind.second.extent;
     break;
    case IndexKind::HYPER:
     for(const auto & ind: hyper_indices) total_extent *= ind.second.extent;
     break;
    default:
     std::cout << "#ERROR(tensor_op_contract:IndexInfo:totalExtent): Invalid index kind!" << std::endl << std::flush;
     assert(false);
   }
   return total_extent;
  }

  void printIt() const{
   std::cout << "IndexInfo{" << std::endl;
   std::cout << " Left indices :";
   for(const auto & lbl: left_indices) std::cout << " " << lbl.first << "["
    << lbl.second.extent << "," << lbl.second.depth << "]";
   std::cout << std::endl;
   std::cout << " Right indices:";
   for(const auto & lbl: right_indices) std::cout << " " << lbl.first << "["
    << lbl.second.extent << "," << lbl.second.depth << "]";
   std::cout << std::endl;
   std::cout << " Contr indices:";
   for(const auto & lbl: contr_indices) std::cout << " " << lbl.first << "["
    << lbl.second.extent << "," << lbl.second.depth << "]";
   std::cout << std::endl;
   std::cout << " Hyper indices:";
   for(const auto & lbl: hyper_indices) std::cout << " " << lbl.first << "["
    << lbl.second.extent << "," << lbl.second.depth << "]";
   std::cout << std::endl;
   std::cout << "}" << std::endl;
   return;
  }

 }; //end struct IndexInfo

 std::shared_ptr<IndexInfo> index_info_;

private:

 bool accumulative_; //accumulative (default) or not
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OP_CONTRACT_HPP_
