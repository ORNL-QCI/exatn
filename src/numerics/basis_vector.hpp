/** ExaTN::Numerics: Basis Vector
REVISION: 2019/02/10

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef BASIS_VECTOR_HPP_
#define BASIS_VECTOR_HPP_

#include "tensor_basic.hpp"

namespace exatn{

namespace numerics{

class BasisVector{
public:

 BasisVector(SubspaceId id = 0);

 BasisVector(const BasisVector & basis_vector) = default;
 BasisVector & operator=(const BasisVector & basis_vector) = default;
 BasisVector(BasisVector && basis_vector) = default;
 BasisVector & operator=(BasisVector && basis_vector) = default;
 virtual ~BasisVector() = default;

 /** Print. **/
 void printIt() const;

private:

 SubspaceId id_; //basis vector id

};

} //namespace numerics

} //namespace exatn

#endif //BASIS_VECTOR_HPP_
