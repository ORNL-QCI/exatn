#ifndef EXATN_RUNTIME_TENSORRUNTIME_HPP_
#define EXATN_RUNTIME_TENSORRUNTIME_HPP_

#include <iostream>
#include <memory>

#include "TensorGraph.hpp"
#include "tensor.hpp"

#include "tensor_method.hpp"

namespace exatn {
namespace runtime {

class TensorRuntime {
protected:
  std::map<std::string, std::shared_ptr<TensorGraph>> dags;
  std::string currentScope;

public:
  void openScope(const std::string &scopeName);
  void closeScope();

  void submit(std::shared_ptr<TensorOperation> op);

  void sync(const std::shared_ptr<TensorOperation> &op);

  void sync(const exatn::numerics::Tensor &tensor);

  TensorDenseBlock getTensorData(const exatn::numerics::Tensor &tensor);
};

} // namespace runtime
} // namespace exatn
#endif
