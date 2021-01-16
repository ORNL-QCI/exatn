/** ExaTN:: Extreme eigenvalue/eigenvector Krylov solver over tensor networks
REVISION: 2021/01/16

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "eigensolver.hpp"

namespace exatn{

unsigned int TensorNetworkEigenSolver::debug{0};


TensorNetworkEigenSolver::TensorNetworkEigenSolver(std::shared_ptr<TensorOperator> tensor_operator,
                                                   std::shared_ptr<TensorExpansion> tensor_expansion,
                                                   double tolerance):
 tensor_operator_(tensor_operator), tensor_expansion_(tensor_expansion),
 max_iterations_(DEFAULT_MAX_ITERATIONS), epsilon_(DEFAULT_LEARN_RATE), tolerance_(tolerance),
 num_roots_(0)
{
}


void TensorNetworkEigenSolver::resetTolerance(double tolerance)
{
 tolerance_ = tolerance;
 return;
}


void TensorNetworkEigenSolver::resetLearningRate(double learn_rate)
{
 epsilon_ = learn_rate;
 return;
}


void TensorNetworkEigenSolver::resetMaxIterations(unsigned int max_iterations)
{
 max_iterations_ = max_iterations;
 return;
}


std::shared_ptr<TensorExpansion> TensorNetworkEigenSolver::getEigenRoot(unsigned int root_id,
                                                                        std::complex<double> * eigenvalue,
                                                                        double * accuracy) const
{
 assert(eigenvalue != nullptr);
 if(root_id >= accuracy_.size()) return std::shared_ptr<TensorExpansion>(nullptr); //invalid root id
 if(accuracy_[root_id] < 0.0) return std::shared_ptr<TensorExpansion>(nullptr); //requested root has not been computed
 if(accuracy != nullptr) *accuracy = accuracy_[root_id];
 *eigenvalue = eigenvalue_[root_id];
 return eigenvector_[root_id];
}


bool TensorNetworkEigenSolver::solve(unsigned int num_roots, const std::vector<double> ** accuracy)
{
 return solve(exatn::getDefaultProcessGroup(),num_roots,accuracy);
}


bool TensorNetworkEigenSolver::solve(const ProcessGroup & process_group,
                                     unsigned int num_roots,
                                     const std::vector<double> ** accuracy)
{
 unsigned int local_rank; //local process rank within the process group
 if(!process_group.rankIsIn(exatn::getProcessRank(),&local_rank)) return true; //process is not in the group: Do nothing

 assert(accuracy != nullptr);
 if(num_roots == 0) return false;
 num_roots_ = num_roots;
 for(unsigned int i = 0; i < num_roots; ++i) accuracy_.emplace_back(-1.0);

 bool success = true;
 //`Finish
 *accuracy = &accuracy_;
 return success;
}


void TensorNetworkEigenSolver::resetDebugLevel(unsigned int level)
{
 TensorNetworkEigenSolver::debug = level;
 return;
}

} //namespace exatn
