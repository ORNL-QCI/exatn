/** ExaTN: MPI Communicator Proxy
REVISION: 2020/05/26

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef EXATN_MPI_COMM_PROXY_HPP_
#define EXATN_MPI_COMM_PROXY_HPP_

#include <vector>

namespace exatn {

class MPICommProxy {
public:

 template<typename MPICommType>
 MPICommProxy(MPICommType * mpi_comm_ptr): mpi_comm_ptr_(static_cast<void*>(mpi_comm_ptr)) {}

 MPICommProxy(): mpi_comm_ptr_(nullptr) {}

 MPICommProxy(const MPICommProxy &) = default;
 MPICommProxy & operator=(const MPICommProxy &) = default;
 MPICommProxy(MPICommProxy &&) noexcept = default;
 MPICommProxy & operator=(MPICommProxy &&) noexcept = default;
 ~MPICommProxy() = default;

 bool isEmpty() const {return (mpi_comm_ptr_ == nullptr);}

 template<typename MPICommType>
 MPICommType * get() const {return static_cast<MPICommType*>(mpi_comm_ptr_);}

 template<typename MPICommType>
 MPICommType & getRef() const {return *(static_cast<MPICommType*>(mpi_comm_ptr_));}

private:
 void * mpi_comm_ptr_; //weak non-owning pointer to an MPI communicator
};


class ProcessGroup {
public:

 static constexpr const std::size_t MAX_MEM_PER_PROCESS = 1UL * 1024UL * 1024UL * 1024UL; //bytes

 ProcessGroup(const MPICommProxy & intra_comm,
              const std::vector<unsigned int> & global_process_ranks):
  process_ranks_(global_process_ranks), intra_comm_(intra_comm), mem_per_process_(MAX_MEM_PER_PROCESS)
 {
  assert(!process_ranks_.empty());
 }

 ProcessGroup(const MPICommProxy & intra_comm,
              const unsigned int group_size):
  process_ranks_(group_size), intra_comm_(intra_comm), mem_per_process_(MAX_MEM_PER_PROCESS)
 {
  assert(process_ranks_.size() > 0);
  for(unsigned int i = 0; i < process_ranks_.size(); ++i) process_ranks_[i] = i;
 }

 ProcessGroup(const ProcessGroup &) = default;
 ProcessGroup & operator=(const ProcessGroup &) = default;
 ProcessGroup(ProcessGroup &&) noexcept = default;
 ProcessGroup & operator=(ProcessGroup &&) noexcept = default;
 ~ProcessGroup() = default;

 unsigned int getSize() const {return process_ranks_.size();}

 const std::vector<unsigned int> & getProcessRanks() const {return process_ranks_;}

 const MPICommProxy & getMPICommProxy() const {return intra_comm_;}

 bool rankIsIn(const unsigned int global_process_rank,
               unsigned int * local_process_rank = nullptr) const
 {
  for(unsigned int i = 0; i < process_ranks_.size(); ++i){
   if(process_ranks_[i] == global_process_rank){
    if(local_process_rank != nullptr) *local_process_rank = i;
    return true;
   }
  }
  return false;
 }

 void resetMemoryLimitPerProcess(std::size_t memory_limit = MAX_MEM_PER_PROCESS){
  mem_per_process_ = memory_limit;
  return;
 }

 std::size_t getMemoryLimitPerProcess() const {return mem_per_process_;}

private:

 std::vector<unsigned int> process_ranks_; //global ranks of the MPI processes forming the process group
 MPICommProxy intra_comm_;                 //associated MPI intra-communicator
 std::size_t mem_per_process_;             //dynamic memory limit per process (bytes)
};

} //namespace exatn

#endif //EXATN_MPI_COMM_PROXY_HPP_
