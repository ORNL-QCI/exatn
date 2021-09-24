/** ExaTN::Numerics: Tensor operation: Contracts two tensors and accumulates the result into another tensor
REVISION: 2021/09/24

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
#include <list>

namespace exatn{

namespace numerics{

class TensorOpContract: public TensorOperation{
public:

 static constexpr double replication_threshold = 0.25; //Range(0..1);

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

 void introduceOptTemporaries(unsigned int num_processes,                        //in: number of processes
                              std::size_t mem_per_process);                      //in: memory limit per process

protected:

 struct Bisect{
  IndexKind index_kind;       //index kind
  unsigned int process_count; //number of associated processes
  bool serial;                //DFS/BFS
 };

 //Information on all indices of a tensor contraction:
 struct IndexInfo{
  std::vector<PosIndexLabel> left_indices_;
  std::vector<PosIndexLabel> right_indices_;
  std::vector<PosIndexLabel> contr_indices_;
  std::vector<PosIndexLabel> hyper_indices_;
  std::map<std::string,const PosIndexLabel*> index_map_; //index name --> pointer to its description
  std::list<Bisect> bisect_sequence_; //pre-calculated sequence of bisections

  //Constructor:
  IndexInfo(const std::vector<PosIndexLabel> & left_indices,   //in: extracted left indices
            const std::vector<PosIndexLabel> & right_indices,  //in: extracted right indices
            const std::vector<PosIndexLabel> & contr_indices,  //in: extracted cotracted indices
            const std::vector<PosIndexLabel> & hyper_indices): //in: extracted hyper indices
   left_indices_(left_indices), right_indices_(right_indices),
   contr_indices_(contr_indices), hyper_indices_(hyper_indices)
  {
   for(const auto & ind: left_indices_){
    auto res = index_map_.emplace(std::make_pair(ind.index_label.label,&ind));
    assert(res.second);
   }
   for(const auto & ind: right_indices_){
    auto res = index_map_.emplace(std::make_pair(ind.index_label.label,&ind));
    assert(res.second);
   }
   for(const auto & ind: contr_indices_){
    auto res = index_map_.emplace(std::make_pair(ind.index_label.label,&ind));
    assert(res.second);
   }
   for(const auto & ind: hyper_indices_){
    auto res = index_map_.emplace(std::make_pair(ind.index_label.label,&ind));
    assert(res.second);
   }
  }

  //Retrieves an index of a tensor contraction:
  const PosIndexLabel * getIndex(const std::string & index_name){
   auto iter = index_map_.find(index_name);
   if(iter == index_map_.end()) return nullptr;
   return iter->second;
  }

  //Prints the index information:
  void printIt() const{
   std::cout << "IndexInfo{" << std::endl;
   std::cout << " Left indices :";
   for(const auto & ind: left_indices_) std::cout << " " << ind.index_label.label
    << "[" << ind.arg_pos[0] << "," << ind.arg_pos[1] << "," << ind.arg_pos[2] << "]" << "{" << ind.depth << "}";
   std::cout << std::endl;
   std::cout << " Right indices:";
   for(const auto & ind: right_indices_) std::cout << " " << ind.index_label.label
    << "[" << ind.arg_pos[0] << "," << ind.arg_pos[1] << "," << ind.arg_pos[2] << "]" << "{" << ind.depth << "}";;
   std::cout << std::endl;
   std::cout << " Contr indices:";
   for(const auto & ind: contr_indices_) std::cout << " " << ind.index_label.label
    << "[" << ind.arg_pos[0] << "," << ind.arg_pos[1] << "," << ind.arg_pos[2] << "]" << "{" << ind.depth << "}";;
   std::cout << std::endl;
   std::cout << " Hyper indices:";
   for(const auto & ind: hyper_indices_) std::cout << " " << ind.index_label.label
    << "[" << ind.arg_pos[0] << "," << ind.arg_pos[1] << "," << ind.arg_pos[2] << "]" << "{" << ind.depth << "}";;
   std::cout << std::endl;
   std::cout << " Bisections:";
   const char bd[2] = {'B','D'};
   for(const auto & bisect: bisect_sequence_)
    std::cout << " " << bd[static_cast<unsigned int>(bisect.serial)] << static_cast<int>(bisect.index_kind)
              << "(" << bisect.process_count << ")";
   std::cout << std::endl;
   std::cout << "}" << std::endl;
  }
 };

 std::shared_ptr<IndexInfo> index_info_; //information on all indices of the tensor contraction

 DimExtent getCombinedDimExtent(IndexKind index_kind) const;

private:

 void determineNumBisections(unsigned int num_processes,
                             std::size_t mem_per_process,
                             unsigned int * bisect_left,
                             unsigned int * bisect_right,
                             unsigned int * bisect_contr,
                             unsigned int * bisect_hyper) const;

 bool accumulative_; //accumulative (default) or not
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OP_CONTRACT_HPP_
