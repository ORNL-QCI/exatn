/** ExaTN: MPI Communicator Proxy
REVISION: 2020/03/10

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef EXATN_MPI_COMM_PROXY_HPP_
#define EXATN_MPI_COMM_PROXY_HPP_

namespace exatn {

class MPICommProxy {
public:

 template<typename MPICommType>
 MPICommProxy(MPICommType * mpi_comm_ptr): mpi_comm_ptr_(static_cast<void*>(mpi_comm_ptr)) {}

 MPICommProxy(const MPICommProxy &) = default;
 MPICommProxy & operator=(const MPICommProxy &) = default;
 MPICommProxy(MPICommProxy &&) noexcept = default;
 MPICommProxy & operator=(MPICommProxy &&) noexcept = default;
 ~MPICommProxy() = default;

 bool isEmpty() const {return (mpi_comm_ptr_ == nullptr);}

 template<typename MPICommType>
 MPICommType * get() const {return static_cast<MPICommType*>(mpi_comm_ptr_);}

private:
 void * mpi_comm_ptr_; //weak non-owning pointer to an MPI communicator
};

} //namespace exatn

#endif //EXATN_MPI_COMM_PROXY_HPP_
