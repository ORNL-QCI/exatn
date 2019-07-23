#ifndef EXATN_RUNTIME_DAGOPT_HPP_
#define EXATN_RUNTIME_DAGOPT_HPP_

#include "TensorGraph.hpp"

#include <iostream>
#include <memory>

namespace exatn {
namespace runtime {

class TensorGraphOptimizer {

public:

   virtual void optimize(TensorGraph & dag) = 0;

};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_DAGOPT_HPP_
