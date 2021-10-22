#ifndef EXATN_HPP_
#define EXATN_HPP_

#include "exatn_service.hpp"
#include "exatn_numerics.hpp"
#include "reconstructor.hpp"
#include "remapper.hpp"
#include "linear_solver.hpp"
#include "optimizer.hpp"
#include "eigensolver.hpp"
#include "param_conf.hpp"

#include "errors.hpp"

namespace exatn {

/** Initializes ExaTN **/
#ifdef MPI_ENABLED
void initialize(const MPICommProxy & communicator,                               //MPI communicator proxy
                const ParamConf & parameters = ParamConf(),                      //runtime configuration parameters
                const std::string & graph_executor_name = "lazy-dag-executor",   //DAG executor kind
                const std::string & node_executor_name = "talsh-node-executor"); //DAG node executor kind
#endif
void initialize(const ParamConf & parameters = ParamConf(),                      //runtime configuration parameters
                const std::string & graph_executor_name = "lazy-dag-executor",   //DAG executor kind
                const std::string & node_executor_name = "talsh-node-executor"); //DAG node executor kind

/** Returns whether or not ExaTN has been initialized **/
bool isInitialized();

/** Finalizes ExaTN **/
void finalize();

} //namespace exatn

#endif //EXATN_HPP_
