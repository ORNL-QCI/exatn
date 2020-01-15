#include "exatn.hpp"

#include <iostream>

namespace exatn {

void initialize() {
  if(!exatnFrameworkInitialized){
    serviceRegistry->initialize();
    exatnFrameworkInitialized = true;
    //std::cout << "#DEBUG(exatn): ExaTN services initialized" << std::endl << std::flush;
    numericalServer = std::make_shared<NumServer>();
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
