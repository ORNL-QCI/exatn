/** ExaTN:: Variational optimizer of a closed symmetric tensor network expansion functional
REVISION: 2021/01/16

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "optimizer.hpp"

namespace exatn{

unsigned int TensorNetworkOptimizer::debug{0};


TensorNetworkOptimizer::TensorNetworkOptimizer(std::shared_ptr<TensorOperator> tensor_operator,
                                               std::shared_ptr<TensorExpansion> vector_expansion,
                                               double tolerance):
 tensor_operator_(tensor_operator), vector_expansion_(vector_expansion),
 max_iterations_(DEFAULT_MAX_ITERATIONS), epsilon_(DEFAULT_LEARN_RATE), tolerance_(tolerance)
{
 if(!vector_expansion_->isKet()){
  std::cout << "#ERROR(exatn:TensorNetworkOptimizer): The tensor network vector expansion must be a ket!"
            << std::endl << std::flush;
  assert(false);
 }
 if(tensor_operator_->getKetRank() != tensor_operator_->getBraRank()){
  std::cout << "#ERROR(exatn:TensorNetworkOptimizer): Tensor operator is not rank-symmetric!"
            << std::endl << std::flush;
  assert(false);
 }
 if(tensor_operator_->getKetRank() != vector_expansion_->getRank()){
  std::cout << "#ERROR(exatn:TensorNetworkOptimizer): Rank mismatch between the tensor operator and the vector!"
            << std::endl << std::flush;
  assert(false);
 }
}


void TensorNetworkOptimizer::resetTolerance(double tolerance)
{
 tolerance_ = tolerance;
 return;
}


void TensorNetworkOptimizer::resetLearningRate(double learn_rate)
{
 epsilon_ = learn_rate;
 return;
}


void TensorNetworkOptimizer::resetMaxIterations(unsigned int max_iterations)
{
 max_iterations_ = max_iterations;
 return;
}


std::shared_ptr<TensorExpansion> TensorNetworkOptimizer::getSolution() const
{
 return vector_expansion_;
}


bool TensorNetworkOptimizer::optimize()
{
 return optimize(exatn::getDefaultProcessGroup());
}

bool TensorNetworkOptimizer::optimize(const ProcessGroup & process_group)
{
 unsigned int local_rank; //local process rank within the process group
 if(!process_group.rankIsIn(exatn::getProcessRank(),&local_rank)) return true; //process is not in the group: Do nothing

 bool success = true;

 if(TensorNetworkOptimizer::debug){
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Tensor network operator:" << std::endl;
  tensor_operator_->printIt();
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Tensor network vector:" << std::endl;
  vector_expansion_->printIt();
 }

 //Construct the operator expectation expansion:
 TensorExpansion bra_vector_expansion(*vector_expansion_);
 bra_vector_expansion.conjugate();
 bra_vector_expansion.rename(vector_expansion_->getName()+"Bra");
 TensorExpansion operator_expectation(bra_vector_expansion,*vector_expansion_,*tensor_operator_);
 operator_expectation.rename("OperatorExpectation");
 if(TensorNetworkOptimizer::debug){
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Operator expectation expansion:" << std::endl;
  operator_expectation.printIt();
 }

 //Construct the metrics expectation expansion:
 TensorExpansion metrics_expectation(bra_vector_expansion,*vector_expansion_);
 metrics_expectation.rename("MetricsExpectation");
 if(TensorNetworkOptimizer::debug){
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Metrics expectation expansion:" << std::endl;
  metrics_expectation.printIt();
 }

 //Construct the residual expectation expansion:
 TensorExpansion residual_expectation;
 success = residual_expectation.appendExpansion(operator_expectation,{1.0,0.0}); assert(success);
 success = residual_expectation.appendExpansion(metrics_expectation,{1.0,0.0}); assert(success);
 residual_expectation.rename("ResidualExpectation");
 if(TensorNetworkOptimizer::debug){
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Residual expectation expansion:" << std::endl;
  residual_expectation.printIt();
 }

 //`Finish
 return success;
}


void TensorNetworkOptimizer::resetDebugLevel(unsigned int level)
{
 TensorNetworkOptimizer::debug = level;
 return;
}

} //namespace exatn
