#include "exatn.hpp"

#include <iostream>

namespace exatn {

#ifdef MPI_ENABLED
void initialize(MPI_Comm communicator,
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
#else
void initialize(const std::string & graph_executor_name,
                const std::string & node_executor_name)
{
  if(!exatnFrameworkInitialized){
    serviceRegistry->initialize();
    exatnFrameworkInitialized = true;
    //std::cout << "#DEBUG(exatn): ExaTN services initialized" << std::endl << std::flush;
    numericalServer = std::make_shared<NumServer>(graph_executor_name,node_executor_name);
    //std::cout << "#DEBUG(exatn): ExaTN numerical server initialized" << std::endl << std::flush;
  }
  return;
}
#endif


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
