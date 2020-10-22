/** ExaTN:: Tensor Runtime: Directed acyclic graph (DAG) of tensor operations
REVISION: 2020/06/23

Copyright (C) 2018-2020 Tiffany Mintz, Dmitry Lyakh, Alex McCaskey
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 (a) The execution space consists of one or more DAGs in which nodes
     represent tensor operations (tasks) and directed edges represent
     dependencies between the corresponding nodes (tensor operations).
     Each DAG is associated with a uniquely named TAProL scope such that
     all tensor operations submitted by the Client to the ExaTN numerical
     server are forwarded into the DAG associated with the TaProL scope
     in which the Client currently resides.
 (b) The tensor graph contains:
     1. The DAG implementation (in the DirectedBoostGraph subclass);
     2. The DAG execution state (TensorExecState data member).
 (c) DEVELOPERS ONLY: The TensorGraph object provides lock/unlock methods for concurrent update
     of the DAG structure (by Client thread) and its execution state (by Execution thread).
     Public virtual methods of TensorGraph implemented in the DirectedBoostGraph subclass
     perform locking/unlocking from there. Other (non-virtual) public methods of TensorGraph
     perform locking/unlocking from here. Additionally each node of the TensorGraph (TensorOpNode object)
     provides more fine grained locking mechanism (lock/unlock methods) for providing exclusive access to
     individual DAG nodes, which is only related to TensorOpNode.getOperation() method since it returns a
     reference to the stored tensor operation (shared pointer reference), thus may require external locking
     for securing an exclusive access to this data member of TensorOpNode.
**/

#ifndef EXATN_RUNTIME_TENSOR_GRAPH_HPP_
#define EXATN_RUNTIME_TENSOR_GRAPH_HPP_

#include "Identifiable.hpp"

#include "tensor_exec_state.hpp"
#include "tensor_operation.hpp"
#include "tensor.hpp"

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>

#include "errors.hpp"

namespace exatn {
namespace runtime {

// Tensor Graph node
class TensorOpNode {

public:
  TensorOpNode():
   op_(nullptr), is_noop_(true), executing_(false), executed_(false), error_(0)
  {}

  TensorOpNode(std::shared_ptr<TensorOperation> tens_op):
   op_(tens_op), is_noop_(false), executing_(false), executed_(false), error_(0)
  {}

  TensorOpNode(const TensorOpNode &) = delete;
  TensorOpNode & operator=(const TensorOpNode &) = delete;
  TensorOpNode(TensorOpNode &&) noexcept = delete;
  TensorOpNode & operator=(TensorOpNode &&) noexcept = delete;
  ~TensorOpNode() = default;

  /** Returns whether or not the TensorOpNode is dummy. **/
  inline bool isDummy() const {return is_noop_;}

  /** Returns a reference to the stored tensor operation. Note that
      this function may require external locking of the TensorOpNode object
      via the lock/unlock methods in order to provide an exclusive access. **/
  inline std::shared_ptr<TensorOperation> & getOperation() {return op_;}

  /** Returns the (unqiue) id of the tensor graph node. **/
  inline VertexIdType getId() const {return id_;}

  /** Returns TRUE if the tensor graph node is currently being executed. **/
  inline bool isExecuting() {return executing_.load();}

  /** Returns TRUE if the tensor graph node has been executed to completion. **/
  inline bool isExecuted(int * error_code = nullptr) {
    bool ans = executed_.load();
    if(error_code != nullptr && ans) *error_code = error_.load();
    return ans;
  }

  /** Returns TRUE if the tensor graph node is neither executed nor currently executing. **/
  inline bool isIdle() {
    bool ans = executing_.load();
    return !(ans || executed_.load());
  }

  /** Sets the (unique) id of the tensor graph node. **/
  inline void setId(VertexIdType id) {
    id_ = id;
    return;
  }

  /** Marks the tensor graph node as being currently executed. **/
  inline void setExecuting() {
    auto executing = executing_.load();
    auto executed = executed_.load();
    if(executing || executed){
      std::cout << "#ERROR(exatn::runtime::TensorOpNode::setExecuting): DAG node not idle: "
                << executing << " " << executed << std::endl;
      if(op_) op_->printIt();
      std::cout << std::flush;
      assert(false);
    }
    executing_.store(true);
    return;
  }

  /** Marks the tensor graph node as executed to completion. **/
  inline void setExecuted(int error_code = 0) {
    auto executing = executing_.load();
    auto executed = executed_.load();
    if(!executing || executed){
      std::cout << "#ERROR(exatn::runtime::TensorOpNode::setExecuted): DAG node is not executing or already executed: "
                << executing << " " << executed << std::endl;
      if(op_) op_->printIt();
      std::cout << std::flush;
      assert(false);
    }
    error_.store(error_code);
    executed_.store(true);
    executing_.store(false);
    return;
  }

  /** Marks the tensor graph node as idle. **/
  inline void setIdle() {
    auto executed = executed_.load();
    if(executed){
      std::cout << "#ERROR(exatn::runtime::TensorOpNode::setIdle): DAG node has been executed to completion: "
                << executing_.load() << " " << executed << std::endl;
      if(op_) op_->printIt();
      std::cout << std::flush;
      assert(false);
    }
    error_.store(0);
    executing_.store(false);
    return;
  }

  inline void lock() {mtx_.lock();}
  inline void unlock() {mtx_.unlock();}

protected:
  std::shared_ptr<TensorOperation> op_; //stored tensor operation
  bool is_noop_;                //TRUE if the stored tensor operation is NOOP (dummy node)
  std::atomic<bool> executing_; //TRUE if the stored tensor operation is currently being executed
  std::atomic<bool> executed_;  //TRUE if the stored tensor operation has been executed to completion
  std::atomic<int> error_;      //execution error code (0:success)
  VertexIdType id_;             //graph vertex id

private:
  std::recursive_mutex mtx_; //object access mutex
};


// Public Tensor Graph API
class TensorGraph : public Identifiable, public Cloneable<TensorGraph> {

public:
  TensorGraph() = default;
  TensorGraph(const TensorGraph &) = delete;
  TensorGraph & operator=(const TensorGraph &) = delete;
  TensorGraph(TensorGraph &&) noexcept = default;
  TensorGraph & operator=(TensorGraph &&) noexcept = default;
  virtual ~TensorGraph() = default;

  /** Adds a new node (tensor operation) into the DAG and returns its id. **/
  virtual VertexIdType addOperation(std::shared_ptr<TensorOperation> op) = 0;

  /** Adds a directed edge between dependent and dependee DAG nodes:
      <dependent> depends on <dependee> (dependent --> dependee). **/
  virtual void addDependency(VertexIdType dependent,
                             VertexIdType dependee) = 0;

  /** Returns TRUE if there is a dependency between two DAG nodes:
      If vertex_id1 node depends on vertex_id2 node. **/
  virtual bool dependencyExists(VertexIdType vertex_id1,
                                VertexIdType vertex_id2) = 0;

  /** Returns the properties (TensorOpNode) of a given DAG node (by reference).
      Subsequently, one may need to lock/unlock the returned TensorOpNode
      in order to ensure a mutually exclusive access to it. **/
  virtual TensorOpNode & getNodeProperties(VertexIdType vertex_id) = 0;

  /** Returns the number of nodes the given node is connected to. **/
  virtual std::size_t getNodeDegree(VertexIdType vertex_id) = 0;

  /** Returns the total number of nodes in the DAG. **/
  virtual std::size_t getNumNodes() = 0;

  /** Returns the total number of dependencies (directed edges) in the DAG. **/
  virtual std::size_t getNumDependencies() = 0;

  /** Returns the list of nodes connected to the given DAG node. **/
  virtual std::vector<VertexIdType> getNeighborList(VertexIdType vertex_id) = 0;

  /** Computes the shortest path from the start index. **/
  virtual void computeShortestPath(VertexIdType startIndex,
                                   std::vector<double> & distances,
                                   std::vector<VertexIdType> & paths) = 0;

  /** Prints the DAG **/
  virtual void printIt() = 0;

  /** Clones an empty subclass instance (needed for plugin registry). **/
  virtual std::shared_ptr<TensorGraph> clone() = 0;


  /** Marks the DAG node as being currently executed. **/
  void setNodeExecuting(VertexIdType vertex_id) {
    return getNodeProperties(vertex_id).setExecuting();
  }

  /** Marks the DAG node as executed to completion. **/
  void setNodeExecuted(VertexIdType vertex_id, int error_code = 0) {
    TensorOpNode & node_properties = getNodeProperties(vertex_id);
    node_properties.setExecuted(error_code);
    auto & op = node_properties.getOperation();
    auto & output_tensor = *(op->getTensorOperand(0)); //`Assumes a single output tensor
    lock();
    auto update_cnt = exec_state_.registerWriteCompletion(output_tensor);
    unlock();
    return;
  }

  /** Marks the DAG node as idle. **/
  void setNodeIdle(VertexIdType vertex_id) {
    return getNodeProperties(vertex_id).setIdle();
  }

  /** Returns TRUE if the DAG node is currently being executed. **/
  bool nodeExecuting(VertexIdType vertex_id) {
    return getNodeProperties(vertex_id).isExecuting();
  }

  /** Returns TRUE if the DAG node has been executed to completion,
      error_code will return the error code (if executed). **/
  bool nodeExecuted(VertexIdType vertex_id, int * error_code = nullptr) {
    return getNodeProperties(vertex_id).isExecuted(error_code);
  }

  /** Returns TRUE if the DAG node is neither executed nor currently executing. **/
  bool nodeIdle(VertexIdType vertex_id) {
    return getNodeProperties(vertex_id).isIdle();
  }

  /** Returns TRUE if all node dependencies have been resolved,
      that is, successfully executed to completion. **/
  bool nodeDependenciesResolved(VertexIdType vertex_id) {
    bool resolved = true;
    lock();
    auto dependencies = getNeighborList(vertex_id);
    for(const auto & dep: dependencies){
      int error_code;
      auto executed = nodeExecuted(dep,&error_code);
      if((!executed) || (error_code != 0)){resolved = false; break;}
    }
    unlock();
    return resolved;
  }

  /** Returns the current outstanding update count on the tensor in the DAG. **/
  inline std::size_t getTensorUpdateCount(const Tensor & tensor) {
    lock();
    auto upd_cnt = exec_state_.getTensorUpdateCount(tensor);
    unlock();
    return upd_cnt;
  }

  /** Registers a DAG node without dependencies. **/
  inline bool registerDependencyFreeNode(VertexIdType node_id) {
    lock();
    auto registered = exec_state_.registerDependencyFreeNode(node_id);
    unlock();
    return registered;
  }

  /** Extracts a dependency-free node from the list.
      Returns FALSE if no such node exists. **/
  inline bool extractDependencyFreeNode(VertexIdType * node_id) {
    lock();
    auto avail = exec_state_.extractDependencyFreeNode(node_id);
    unlock();
    return avail;
  }

  /** Returns the current list of dependency free nodes. **/
  inline std::list<VertexIdType> getDependencyFreeNodes() {
    lock();
    auto nodes = exec_state_.getDependencyFreeNodes();
    unlock();
    return std::move(nodes);
  }

  /** Registers a DAG node as being executed (together with its execution handle). **/
  inline void registerExecutingNode(VertexIdType node_id,
                                    TensorOpExecHandle exec_handle) {
    lock();
    exec_state_.registerExecutingNode(node_id,exec_handle);
    unlock();
    return;
  }

  /** Extracts an executed DAG node from the list of executing nodes. **/
  inline bool extractExecutingNode(VertexIdType * node_id) {
    lock();
    auto avail = exec_state_.extractExecutingNode(node_id);
    unlock();
    return avail;
  }

  inline ExecutingNodesIterator extractExecutingNode(ExecutingNodesIterator node_iterator,
                                                     VertexIdType * node_id) {
    lock();
    auto new_iter = exec_state_.extractExecutingNode(node_iterator,node_id);
    unlock();
    return new_iter;
  }

  /** Returns a constant iterator to the list of currently executing DAG nodes. **/
  inline ExecutingNodesIterator executingNodesBegin() const {return exec_state_.executingNodesBegin();}
  inline ExecutingNodesIterator executingNodesEnd() const {return exec_state_.executingNodesEnd();}

  /** Given just executed DAG node, moves forward the DAG front node
      if appropriate. **/
  inline bool progressFrontNode(VertexIdType node_executed) {
    return exec_state_.progressFrontNode(node_executed);
  }

  /** Returns the current front node id. **/
  inline VertexIdType getFrontNode() const {
    return exec_state_.getFrontNode();
  }

  /** Affirms that the DAG has unexecuted nodes. **/
  inline bool hasUnexecutedNodes() {
    return (exec_state_.getFrontNode() < this->getNumNodes());
  }

  inline void lock() {mtx_.lock();}
  inline void unlock() {mtx_.unlock();}

protected:
  TensorExecState exec_state_; //tensor graph execution state

private:
  std::recursive_mutex mtx_; //object access mutex
};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_TENSOR_GRAPH_HPP_
