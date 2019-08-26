#include "exatn.hpp"

namespace exatn {

void initialize() {
  if(!exatnFrameworkInitialized){
    serviceRegistry->initialize();
    numericalServer = std::make_shared<NumServer>();
    exatnFrameworkInitialized = true;
  }
  return;
}


bool isInitialized() {
  return exatnFrameworkInitialized;
}


void finalize() {
  exatnFrameworkInitialized = false;
  numericalServer.reset();
  return;
}

} // namespace exatn
