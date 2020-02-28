#ifndef EXATN_HPP_
#define EXATN_HPP_

#include "exatn_service.hpp"
#include "exatn_numerics.hpp"
#include "reconstructor.hpp"
#include "optimizer.hpp"
#include "eigensolver.hpp"

namespace exatn {

/** Initializes ExaTN **/
#ifdef MPI_ENABLED
void initialize(MPI_Comm communicator = MPI_COMM_WORLD,                          //MPI communicator
                const std::string & graph_executor_name = "eager-dag-executor",  //DAG executor kind
                const std::string & node_executor_name = "talsh-node-executor"); //DAG node executor kind
#else
void initialize(const std::string & graph_executor_name = "eager-dag-executor",  //DAG executor kind
                const std::string & node_executor_name = "talsh-node-executor"); //DAG node executor kind
#endif

/** Returns whether or not ExaTN has been initialized **/
bool isInitialized();

/** Finalizes ExaTN **/
void finalize();

} //namespace exatn

#endif //EXATN_HPP_
