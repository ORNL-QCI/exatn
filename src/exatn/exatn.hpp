#ifndef EXATN_HPP_
#define EXATN_HPP_

#include "exatn_service.hpp"
#include "exatn_numerics.hpp"
#include "reconstructor.hpp"
#include "optimizer.hpp"
#include "eigensolver.hpp"

namespace exatn {

/** Initializes ExaTN **/
void initialize();

/** Returns whether or not ExaTN has been initialized **/
bool isInitialized();

/** Finalizes ExaTN **/
void finalize();

} //namespace exatn

#endif //EXATN_HPP_
