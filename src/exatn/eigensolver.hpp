/** ExaTN:: Extreme eigenvalue/vector solver over tensor networks
REVISION: 2019/12/18

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Finds the approximate extreme eigenvalues and corresponding eigenvectors
     expanded in the Krylov subspace spanned by tensor networks using the
     Davidson-Nakatsuji-Hirao algorithm for non-Hermitian tensor operators.
**/

#ifndef EXATN_EIGENSOLVER_HPP_
#define EXATN_EIGENSOLVER_HPP_

#include "exatn_numerics.hpp"
#include "reconstructor.hpp"
#include "optimizer.hpp"

#include <vector>
#include <complex>
#include <memory>

namespace exatn{

class TensorNetworkEigenSolver{

public:

 TensorNetworkEigenSolver(std::shared_ptr<TensorOperator> tensor_operator,   //in: tensor operator the extreme eigenroots of which need to be found
                          std::shared_ptr<TensorExpansion> tensor_expansion, //in: tensor expansion form that will be used for each eigenvector
                          double tolerance);                                 //in: desired numerical covergence tolerance

 TensorNetworkEigenSolver(const TensorNetworkEigenSolver &) = default;
 TensorNetworkEigenSolver & operator=(const TensorNetworkEigenSolver &) = default;
 TensorNetworkEigenSolver(TensorNetworkEigenSolver &&) noexcept = default;
 TensorNetworkEigenSolver & operator=(TensorNetworkEigenSolver &&) noexcept = default;
 ~TensorNetworkEigenSolver() = default;

 /** Runs the eigensolver for one or more extreme eigenroots.
     Upon success, returns the achieved accuracy for each eigenroot. **/
 bool solve(unsigned int num_roots,                 //in: number of extreme eigenroots to find
            const std::vector<double> ** accuracy); //out: achieved accuracy for each root: accuracy[num_roots]

 /** Returns the requested eigenvalue and eigenvector. **/
 std::shared_ptr<TensorExpansion> getEigenRoot(unsigned int root_id,              //in: root id: [0..max]
                                               std::complex<double> * eigenvalue, //out: eigenvalue
                                               double * accuracy = nullptr);      //out: achieved accuracy

private:

 std::shared_ptr<TensorOperator> tensor_operator_;           //tensor operator the extreme eigenroots of which need to be found
 std::shared_ptr<TensorExpansion> tensor_expansion_;         //desired form of the eigenvector as a tensor expansion
 std::vector<std::shared_ptr<TensorExpansion>> eigenvector_; //tensor expansion approximating each requested eigenvector
 std::vector<std::complex<double>> eigenvalue_;              //computed eigenvalues
 std::vector<double> accuracy_;                              //actually achieved accuracy for each eigenroot
 double tolerance_;                                          //desired numerical convergence tolerance
 unsigned int num_roots_;                                    //number of extreme eigenroots requested
};

} //namespace exatn

#endif //EXATN_EIGENSOLVER_HPP_
