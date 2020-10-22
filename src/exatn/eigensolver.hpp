/** ExaTN:: Extreme eigenvalue/eigenvector solver over tensor networks
REVISION: 2020/01/24

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) The tensor network expansion eigensolver finds the approximate extreme
     eigenvalues and their corresponding eigenvectors expanded in the Krylov
     subspace spanned by tensor network expansions. The procedure is derived
     from the Davidson-Nakatsuji-Hirao algorithm for non-Hermitian matrices,
     which in turn is based on the Arnoldi algorithm.
**/

#ifndef EXATN_EIGENSOLVER_HPP_
#define EXATN_EIGENSOLVER_HPP_

#include "exatn_numerics.hpp"
#include "reconstructor.hpp"
#include "optimizer.hpp"

#include <vector>
#include <complex>
#include <memory>

#include "errors.hpp"

namespace exatn{

class TensorNetworkEigenSolver{

public:

 TensorNetworkEigenSolver(std::shared_ptr<TensorOperator> tensor_operator,   //in: tensor operator the extreme eigenroots of which are to be found
                          std::shared_ptr<TensorExpansion> tensor_expansion, //in: tensor network expansion form that will be used for each eigenvector
                          double tolerance);                                 //in: desired numerical covergence tolerance

 TensorNetworkEigenSolver(const TensorNetworkEigenSolver &) = default;
 TensorNetworkEigenSolver & operator=(const TensorNetworkEigenSolver &) = default;
 TensorNetworkEigenSolver(TensorNetworkEigenSolver &&) noexcept = default;
 TensorNetworkEigenSolver & operator=(TensorNetworkEigenSolver &&) noexcept = default;
 ~TensorNetworkEigenSolver() = default;

 /** Runs the tensor network eigensolver for one or more extreme eigenroots
     of the underlying tensor operator. Upon success, returns the achieved
     accuracy for each eigenroot. **/
 bool solve(unsigned int num_roots,                 //in: number of extreme eigenroots to find
            const std::vector<double> ** accuracy); //out: achieved accuracy for each root: accuracy[num_roots]

 /** Returns the requested eigenvalue and eigenvector. **/
 std::shared_ptr<TensorExpansion> getEigenRoot(unsigned int root_id,              //in: root id: [0..max]
                                               std::complex<double> * eigenvalue, //out: eigenvalue
                                               double * accuracy = nullptr);      //out: achieved accuracy

private:

 std::shared_ptr<TensorOperator> tensor_operator_;           //tensor operator the extreme eigenroots of which are to be found
 std::shared_ptr<TensorExpansion> tensor_expansion_;         //desired form of the eigenvector as a tensor network expansion
 std::vector<std::shared_ptr<TensorExpansion>> eigenvector_; //tensor network expansion approximating each requested eigenvector
 std::vector<std::complex<double>> eigenvalue_;              //computed eigenvalues
 std::vector<double> accuracy_;                              //actually achieved accuracy for each eigenroot
 double tolerance_;                                          //desired numerical convergence tolerance for each eigenroot
 unsigned int num_roots_;                                    //number of extreme eigenroots requested
};

} //namespace exatn

#endif //EXATN_EIGENSOLVER_HPP_
