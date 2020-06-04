/** ExaTN: MPI Communicator Proxy & Process group
REVISION: 2020/06/03

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "mpi_proxy.hpp"

#ifdef MPI_ENABLED
#include "mpi.h"
#endif

#include <cstdlib>

#include <iostream>

namespace exatn {

void MPICommProxy::allocateMPICommPtr(std::size_t bytes)
{
 mpi_comm_ptr_ = malloc(bytes);
 return;
}


MPICommProxy::~MPICommProxy()
{
 if(mpi_comm_ptr_ != nullptr) free(mpi_comm_ptr_);
}


ProcessGroup::~ProcessGroup()
{
#ifdef MPI_ENABLED
 if(!intra_comm_.isEmpty()){
  auto * mpicomm = intra_comm_.get<MPI_Comm>();
  if(*mpicomm != MPI_COMM_WORLD && *mpicomm != MPI_COMM_SELF){
   auto errc = MPI_Comm_free(mpicomm); assert(errc == MPI_SUCCESS);
  }
 }
#endif
}


std::shared_ptr<ProcessGroup> ProcessGroup::split(int my_subgroup) const
{
 if(this->getSize() == 1) return std::make_shared<ProcessGroup>(*this); //cannot split single-process group
 std::shared_ptr<ProcessGroup> subgroup;
#ifdef MPI_ENABLED
 if(!intra_comm_.isEmpty()){
  auto * mpicomm = intra_comm_.get<MPI_Comm>();
  int color = MPI_UNDEFINED;
  if(my_subgroup >= 0) color = my_subgroup;
  MPICommProxy subgroup_comm(mpicomm);
  auto * subgroup_mpicomm = subgroup_comm.get<MPI_Comm>();
  auto errc = MPI_Comm_split(*mpicomm,color,0,subgroup_mpicomm); assert(errc == MPI_SUCCESS);
  //`Retrieve MPI process ranks forming the subgroup and finish subgroup construction (+ copy memory limit)
 }else{
  std::cout << "#ERROR(exatn::ProcessGroup::split): Empty MPI communicator!\n" << std::flush;
  assert(false);
 }
#endif
 return subgroup;
}

} //namespace exatn
