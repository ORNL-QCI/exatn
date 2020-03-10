/** ExaTN: MPI proxy types
REVISION: 2020/03/10

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef EXATN_MPI_PROXY_HPP_
#define EXATN_MPI_PROXY_HPP_

namespace exatn {

class MPICommProxy{
public:

 template<typename MPITypeName>
 MPICommProxy(MPITypeName * mpi_object_ptr):
  object_(static_cast<void*>(mpi_object_ptr)) {}

 MPICommProxy(const MPICommProxy &) = default;
 MPICommProxy & operator=(const MPICommProxy &) = default;
 MPICommProxy(MPICommProxy &&) noexcept = default;
 MPICommProxy & operator=(MPICommProxy &&) noexcept = default;
 ~MPICommProxy() = default;

 bool isEmpty() const {return (object_ == nullptr);}

 template<typename MPITypeName>
 MPITypeName * get(){return static_cast<MPITypeName*>(object_);}

private:
 void * object_;
};

} //namespace exatn

#endif //EXATN_MPI_PROXY_HPP_
