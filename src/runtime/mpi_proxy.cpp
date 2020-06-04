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

ProcessGroup::~ProcessGroup()
{
#ifdef MPI_ENABLED
 if(!intra_comm_.isEmpty()){
  auto * mpicomm = intra_comm_.get<MPI_Comm>();
  int res;
  auto errc = MPI_Comm_compare(*mpicomm,MPI_COMM_WORLD,&res); assert(errc == MPI_SUCCESS);
  if(res != MPI_IDENT){
   errc = MPI_Comm_compare(*mpicomm,MPI_COMM_SELF,&res); assert(errc == MPI_SUCCESS);
   if(res != MPI_IDENT){
    errc = MPI_Comm_free(mpicomm); assert(errc == MPI_SUCCESS);
   }
  }
 }
#endif
}


std::shared_ptr<ProcessGroup> ProcessGroup::split(int my_subgroup) const
{
 if(this->getSize() == 1) return std::make_shared<ProcessGroup>(*this); //cannot split single-process group
 std::shared_ptr<ProcessGroup> subgroup(nullptr);
#ifdef MPI_ENABLED
 if(!intra_comm_.isEmpty()){
  auto & mpicomm = intra_comm_.getRef<MPI_Comm>();
  int color = MPI_UNDEFINED;
  if(my_subgroup >= 0) color = my_subgroup;
  MPI_Comm subgroup_mpicomm;
  auto errc = MPI_Comm_split(mpicomm,color,0,&subgroup_mpicomm); assert(errc == MPI_SUCCESS);
  if(color != MPI_UNDEFINED){
   int subgroup_size;
   errc = MPI_Comm_size(subgroup_mpicomm,&subgroup_size); assert(errc == MPI_SUCCESS);
   MPI_Group orig_group,new_group;
   errc = MPI_Comm_group(mpicomm,&orig_group); assert(errc == MPI_SUCCESS);
   errc = MPI_Comm_group(subgroup_mpicomm,&new_group); assert(errc == MPI_SUCCESS);
   int sub_ranks[subgroup_size],orig_ranks[subgroup_size];
   for(int i = 0; i < subgroup_size; ++i) sub_ranks[i] = i;
   errc = MPI_Group_translate_ranks(new_group,subgroup_size,sub_ranks,orig_group,orig_ranks);
   std::vector<unsigned int> subgroup_ranks(subgroup_size);
   const auto & ranks = this->getProcessRanks();
   for(int i = 0; i < subgroup_size; ++i) subgroup_ranks[i] = ranks[orig_ranks[i]];
   subgroup = std::make_shared<ProcessGroup>(MPICommProxy(subgroup_mpicomm),subgroup_ranks,this->getMemoryLimitPerProcess());
  }
 }else{
  std::cout << "#ERROR(exatn::ProcessGroup::split): Empty MPI communicator!\n" << std::flush;
  assert(false);
 }
#endif
 return subgroup;
}

} //namespace exatn
