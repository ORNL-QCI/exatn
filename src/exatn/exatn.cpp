#include "exatn.hpp"

#ifdef MPI_ENABLED
#include "mpi.h"
#endif

#include <iostream>

namespace exatn {

#ifdef MPI_ENABLED
void initialize(MPICommProxy & communicator,
                const std::string & graph_executor_name,
                const std::string & node_executor_name)
{
  if(!exatnFrameworkInitialized){
    serviceRegistry->initialize();
    exatnFrameworkInitialized = true;
    //std::cout << "#DEBUG(exatn): ExaTN services initialized" << std::endl << std::flush;
    numericalServer = std::make_shared<NumServer>(communicator,graph_executor_name,node_executor_name);
    //std::cout << "#DEBUG(exatn): ExaTN numerical server initialized" << std::endl << std::flush;
  }
  return;
}
#endif


void initialize(const std::string & graph_executor_name,
                const std::string & node_executor_name)
{
  if(!exatnFrameworkInitialized){
    serviceRegistry->initialize();
    exatnFrameworkInitialized = true;
    //std::cout << "#DEBUG(exatn): ExaTN services initialized" << std::endl << std::flush;
#ifdef MPI_ENABLED
    numericalServer = std::make_shared<NumServer>(???,graph_executor_name,node_executor_name);
#else
    numericalServer = std::make_shared<NumServer>(graph_executor_name,node_executor_name);
#endif
    //std::cout << "#DEBUG(exatn): ExaTN numerical server initialized" << std::endl << std::flush;
  }
  return;
}


bool isInitialized() {
  return exatnFrameworkInitialized;
}


void finalize() {
  numericalServer.reset();
  exatnFrameworkInitialized = false;
  //std::cout << "#DEBUG(exatn): ExaTN numerical server shut down" << std::endl << std::flush;
  return;
}

} // namespace exatn
