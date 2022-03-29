/** ExaTN:: Tensor Runtime: Task-based execution layer for tensor operations
REVISION: 2022/03/29

Copyright (C) 2018-2022 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 (a) The execution space consists of one or more DAGs in which nodes
     represent tensor operations (tasks) and directed edges represent
     dependencies between the corresponding nodes (tensor operations).
     Each DAG is associated with a uniquely named TAProL scope such that
     all tensor operations submitted by the Client to the ExaTN numerical
     server are forwarded into the DAG associated with the TaProL scope
     in which the Client currently resides.
 (b) The DAG lifecycle:
     openScope(name): Opens a new TAProL scope and creates its associated empty DAG.
                      The .submit method can then be used to append new tensor
                      operations or whole tensor networks into the current DAG.
                      The actual execution of the submitted tensor operations
                      is asynchronous and may start any time after submission.
     pauseScope(): Completes the actual execution of all started tensor operations in the
                   current DAG and defers the execution of the rest of the DAG for later.
     resumeScope(name): Pauses the execution of the currently active DAG (if any) and
                        resumes the execution of a previously paused DAG, making it current.
     closeScope(): Completes all tensor operations in the current DAG and destroys it.
 (c) submit(TensorOperation): Submits a tensor operation for (generally deferred) execution.
     sync(TensorOperation): Tests for completion of a specific tensor operation.
     sync(tensor): Tests for completion of all submitted update operations on a given tensor.
 (d) Upon creation, the TensorRuntime object spawns an execution thread which will be executing tensor
     operations in the course of DAG traversal. The execution thread will be joined upon TensorRuntime
     destruction. After spawning the execution thread, the main thread returns control to the client
     which will then be able to submit new operations into the current DAG. The submitted operations
     will be autonomously executed by the execution thread. The DAG execution policy is specified by
     a polymorphic TensorGraphExecutor provided during the construction of the TensorRuntime.
     Correspondingly, the TensorGraphExecutor contains a polymorphic TensorNodeExecutor responsible
     for the actual execution of submitted tensor operations via an associated computational backend.
     The concrete TensorNodeExecutor is specified during the construction of the TensorRuntime oject.
 (e) DEVELOPERS ONLY: The TensorGraph object (DAG) provides lock/unlock methods for concurrent update
     of the DAG structure (by Client thread) and its execution state (by Execution thread).
     Additionally each node of the TensorGraph (TensorOpNode object) provides more fine grain
     locking mechanism (lock/unlock methods) for providing exclusive access to individual DAG nodes.
**/

#ifndef EXATN_RUNTIME_TENSOR_RUNTIME_HPP_
#define EXATN_RUNTIME_TENSOR_RUNTIME_HPP_

#include "tensor_graph.hpp"
#include "tensor_network_queue.hpp"
#include "tensor_graph_executor.hpp"
#include "tensor_operation.hpp"
#include "tensor_method.hpp"

#include "param_conf.hpp"
#include "mpi_proxy.hpp"

#include <map>
#include <list>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <future>
#include <mutex>

namespace exatn {
namespace runtime {

enum class CompBackend {
  None
 ,Default
#ifdef CUQUANTUM
 ,Cuquantum
#endif
};

class TensorRuntime final {

public:

  static constexpr std::size_t MAX_RUNTIME_DAG_SIZE = 8192; //max allowed DAG size during runtime

#ifdef MPI_ENABLED
  TensorRuntime(const MPICommProxy & communicator,                               //MPI communicator proxy
                const ParamConf & parameters,                                    //runtime configuration parameters
                const std::string & graph_executor_name = "eager-dag-executor",  //DAG executor kind
                const std::string & node_executor_name = "talsh-node-executor"); //DAG node executor kind
#else
  TensorRuntime(const ParamConf & parameters,                                    //runtime configuration parameters
                const std::string & graph_executor_name = "eager-dag-executor",  //DAG executor kind
                const std::string & node_executor_name = "talsh-node-executor"); //DAG node executor kind
#endif
  TensorRuntime(const TensorRuntime &) = delete;
  TensorRuntime & operator=(const TensorRuntime &) = delete;
  TensorRuntime(TensorRuntime &&) noexcept = delete;
  TensorRuntime & operator=(TensorRuntime &&) noexcept = delete;
  ~TensorRuntime();

  /** Resets the logging level (0:none) [MAIN THREAD]. **/
  void resetLoggingLevel(int level = 0);

  /** Enforces serialized (synchronized) execution of the DAG. **/
  void resetSerialization(bool serialize,
                          bool validation_trace = false);

  /** Activates/deactivates dry run (no actual computations). **/
  void activateDryRun(bool dry_run);

  /** Activates mixed-precision fast math on all devices (if available). **/
  void activateFastMath();

  /** Returns the Host memory buffer size in bytes provided by the executor. **/
  std::size_t getMemoryBufferSize() const;

  /** Returns the current memory usage by all allocated tensors.
      Note that the returned value includes buffer fragmentation overhead. **/
  std::size_t getMemoryUsage(std::size_t * free_mem) const;

  /** Returns the current value of the total Flop count executed by the executor. **/
  double getTotalFlopCount() const;

  /** Opens a new scope represented by a new execution graph (DAG). **/
  void openScope(const std::string & scope_name);

  /** Pauses the current scope by completing all outstanding tensor operations
      and pausing the further progress of the current execution graph until resume.
      Returns TRUE upon successful pausing, FALSE otherwise. **/
  void pauseScope();

  /** Resumes the execution of a previously paused scope (execution graph). **/
  void resumeScope(const std::string & scope_name);

  /** Closes the current scope, fully completing all tensor operations
      in the current execution graph. **/
  void closeScope();

  /** Returns TRUE if there is the current scope is set. **/
  inline bool currentScopeIsSet() const {return scope_set_.load();}

  /** Submits a tensor operation into the current execution graph and returns its integer id. **/
  VertexIdType submit(std::shared_ptr<TensorOperation> op); //in: tensor operation

  /** Tests for completion of a given tensor operation.
      If wait = TRUE, it will block until completion. **/
  bool sync(TensorOperation & op,
            bool wait = true);

  /** Tests for completion of all outstanding update operations on a given tensor.
      If wait = TRUE, it will block until completion. **/
  bool sync(const Tensor & tensor,
            bool wait = true);

  /** Tests for completion of all previously submitted tensor operations.
      If wait = TRUE, it will block until completion. **/
  bool sync(bool wait = true);

#ifdef CUQUANTUM
  /** Submits an entire tensor network for processing as a whole.
      The returned execution handle can be used for checking the status
      of the tensor network execution. Zero on return means unsuccessful submission. **/
  TensorOpExecHandle submit(std::shared_ptr<numerics::TensorNetwork> network, //in: tensor network
                            const MPICommProxy & communicator, //MPI communicator proxy
                            unsigned int num_processes, //in: number of executing processes
                            unsigned int process_rank); //in: rank of the current executing process

  /** Tests for completion of processing of a whole tensor network.
      A valid execution handle obtained during tensor network
      submission must be positive. **/
  bool syncNetwork(const TensorOpExecHandle exec_handle,
                   bool wait = true);
#endif

  /** Returns a locally stored tensor slice (talsh::Tensor) providing access to tensor elements.
      This slice will be extracted from the exatn::numerics::Tensor implementation as a copy.
      The returned future becomes ready once the execution thread has retrieved the slice copy. **/
  std::future<std::shared_ptr<talsh::Tensor>> getLocalTensor(std::shared_ptr<Tensor> tensor, //in: exatn::numerics::Tensor to get slice of (by copy)
                            const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec); //in: tensor slice specification

private:
  /** Tensor data request **/
  class TensorDataReq{
  public:
   std::promise<std::shared_ptr<talsh::Tensor>> slice_promise_;
   std::vector<std::pair<DimOffset,DimExtent>> slice_specs_;
   std::shared_ptr<Tensor> tensor_;

   TensorDataReq(std::promise<std::shared_ptr<talsh::Tensor>> && slice_promise,
                 const std::vector<std::pair<DimOffset,DimExtent>> & slice_specs,
                 std::shared_ptr<Tensor> tensor):
    slice_promise_(std::move(slice_promise)), slice_specs_(slice_specs), tensor_(tensor){}

   TensorDataReq(const TensorDataReq & req) = delete;
   TensorDataReq & operator=(const TensorDataReq & req) = delete;
   TensorDataReq(TensorDataReq && req) noexcept = default;
   TensorDataReq & operator=(TensorDataReq && req) noexcept = default;
   ~TensorDataReq() = default;
  };

  /** Launches the execution thread which will be executing DAGs on the fly. **/
  void launchExecutionThread();
  /** The execution thread lives here. **/
  void executionThreadWorkflow();
  /** Processes all outstanding tensor data requests (by execution thread). **/
  void processTensorDataRequests();

  inline void lockDataReqQ(){data_req_mtx_.lock();}
  inline void unlockDataReqQ(){data_req_mtx_.unlock();}

  /** Runtime configuration parameters **/
  ParamConf parameters_;
  /** Tensor graph (DAG) executor name **/
  std::string graph_executor_name_;
  /** Tensor graph (DAG) node executor name **/
  std::string node_executor_name_;
  /** Total number of parallel processes in the dedicated MPI communicator **/
  int num_processes_;
  /** Rank of the current parallel process in the dedicated MPI communicator **/
  int process_rank_;
  /** Rank of the current parallel process in MPI_COMM_WORLD **/
  int global_process_rank_;
  /** Current tensor graph (DAG) executor **/
  std::shared_ptr<TensorGraphExecutor> graph_executor_;
  /** Active execution graphs (DAGs) **/
  std::map<std::string, std::shared_ptr<TensorGraph>> dags_;
  /** Name of the current scope (current DAG name) **/
  std::string current_scope_;
  /** Current DAG **/
  std::shared_ptr<TensorGraph> current_dag_; //pointer to the current DAG
  /** Tensor data request queue **/
  std::list<TensorDataReq> data_req_queue_;
  /** List of tensor networks submitted for processing as a whole **/
  TensorNetworkQueue tensor_network_queue_;
  /** Logging level (0:none) **/
  int logging_;
  /** Currently used computational backend **/
  std::atomic<CompBackend> backend_;
  /** Current executing status (whether or not the execution thread is active) **/
  std::atomic<bool> executing_; //TRUE while the execution thread is executing the current DAG
  /** Current scope status **/
  std::atomic<bool> scope_set_; //TRUE if the current scope is set
  /** End of life flag **/
  std::atomic<bool> alive_; //TRUE while the main thread is accepting new operations from Client
  /** Execution thread **/
  std::thread exec_thread_;
  /** Data request mutex **/
  std::mutex data_req_mtx_;
};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_TENSOR_RUNTIME_HPP_
