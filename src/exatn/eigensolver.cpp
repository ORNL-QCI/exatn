/** ExaTN:: Extreme eigenvalue/eigenvector solver over tensor networks
REVISION: 2020/01/24

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "eigensolver.hpp"

#include <cassert>

namespace exatn{

TensorNetworkEigenSolver::TensorNetworkEigenSolver(std::shared_ptr<TensorOperator> tensor_operator,
                                                   std::shared_ptr<TensorExpansion> tensor_expansion,
                                                   double tolerance):
 tensor_operator_(tensor_operator), tensor_expansion_(tensor_expansion), tolerance_(tolerance), num_roots_(0)
{
}


std::shared_ptr<TensorExpansion> TensorNetworkEigenSolver::getEigenRoot(unsigned int root_id,
                                                                        std::complex<double> * eigenvalue,
                                                                        double * accuracy)
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
 assert(accuracy != nullptr);
 if(num_roots == 0) return false;
 num_roots_ = num_roots;
 for(unsigned int i = 0; i < num_roots; ++i) accuracy_.emplace_back(-1.0);
 //`Finish
 *accuracy = &accuracy_;
 return true;
}

} //namespace exatn
