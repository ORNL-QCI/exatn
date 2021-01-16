/** ExaTN: Extreme eigenvalue/eigenvector Krylov solver over tensor networks
REVISION: 2021/01/16

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) The tensor network expansion eigensolver finds approximate extreme
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

#include "errors.hpp"

namespace exatn{

class TensorNetworkEigenSolver{

public:

 static unsigned int debug;

 static constexpr const double DEFAULT_TOLERANCE = 1e-5;
 static constexpr const double DEFAULT_LEARN_RATE = 0.5;
 static constexpr const unsigned int DEFAULT_MAX_ITERATIONS = 1000;

 TensorNetworkEigenSolver(std::shared_ptr<TensorOperator> tensor_operator,   //in: tensor operator the extreme eigenroots of which are to be found
                          std::shared_ptr<TensorExpansion> tensor_expansion, //in: tensor network expansion form that will be used for each eigenvector
                          double tolerance = DEFAULT_TOLERANCE);             //in: desired numerical convergence tolerance

 TensorNetworkEigenSolver(const TensorNetworkEigenSolver &) = default;
 TensorNetworkEigenSolver & operator=(const TensorNetworkEigenSolver &) = default;
 TensorNetworkEigenSolver(TensorNetworkEigenSolver &&) noexcept = default;
 TensorNetworkEigenSolver & operator=(TensorNetworkEigenSolver &&) noexcept = default;
 ~TensorNetworkEigenSolver() = default;

 /** Resets the numerical tolerance. **/
 void resetTolerance(double tolerance = DEFAULT_TOLERANCE);

 /** Resets the learning rate. **/
 void resetLearningRate(double learn_rate = DEFAULT_LEARN_RATE);

 /** Resets the max number of macro-iterations. **/
 void resetMaxIterations(unsigned int max_iterations = DEFAULT_MAX_ITERATIONS);

 /** Runs the tensor network eigensolver for one or more extreme eigenroots
     of the underlying tensor operator. Upon success, returns the achieved
     accuracy for each eigenroot. **/
 bool solve(unsigned int num_roots,                 //in: number of extreme eigenroots to find
            const std::vector<double> ** accuracy); //out: achieved accuracy for each root: accuracy[num_roots]
 bool solve(const ProcessGroup & process_group,     //in: executing process group
            unsigned int num_roots,                 //in: number of extreme eigenroots to find
            const std::vector<double> ** accuracy); //out: achieved accuracy for each root: accuracy[num_roots]

 /** Returns the requested eigenvalue and eigenvector. **/
 std::shared_ptr<TensorExpansion> getEigenRoot(unsigned int root_id,               //in: root id: [0..max]
                                               std::complex<double> * eigenvalue,  //out: eigenvalue
                                               double * accuracy = nullptr) const; //out: achieved accuracy

 static void resetDebugLevel(unsigned int level = 0);

private:

 std::shared_ptr<TensorOperator> tensor_operator_;           //tensor operator the extreme eigenroots of which are to be found
 std::shared_ptr<TensorExpansion> tensor_expansion_;         //desired form of the eigenvector as a tensor network expansion
 unsigned int max_iterations_;                               //max number of macro-iterations
 double epsilon_;                                            //learning rate for the gradient descent based tensor update
 double tolerance_;                                          //numerical convergence tolerance (for the gradient)

 unsigned int num_roots_;                                    //number of extreme eigenroots requested
 std::vector<std::shared_ptr<TensorExpansion>> eigenvector_; //tensor network expansion approximating each requested eigenvector
 std::vector<std::complex<double>> eigenvalue_;              //computed eigenvalues
 std::vector<double> accuracy_;                              //actually achieved accuracy for each eigenroot
};

} //namespace exatn

#endif //EXATN_EIGENSOLVER_HPP_
