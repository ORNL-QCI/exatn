#ifndef EXATN_RUNTIME_DAGOPT_HPP_
#define EXATN_RUNTIME_DAGOPT_HPP_

#include <iostream>
#include <memory>

namespace exatn {
namespace runtime {

class GraphOptimizer {

public:

   virtual void optimize(Graph& dag) = 0;

};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_DAGOPT_HPP_
