
#include <vector>

[[clang::syntax(taprol)]] void test(std::vector<std::complex<float>> t2,
                                    std::shared_ptr<talsh::Tensor> talsh_t2,
                                    std::shared_ptr<talsh::Tensor> talsh_x2,
                                    double &norm_x2) {
entry: main;
  scope main group();
  space(complex) : my_space = [0:255], your_space = [0:511];
  subspace(my_space) : s0 = [0:127], s1 = [128:255];
  subspace(your_space) : r0 = [42:84], r1 = [484:511];
  index(s0) : i, j, k, l;
  index(r0) : a, b, c, d;
  H2(a, i, b, j) = {0.0, 0.0};
  H2(a, i, b, j) = method("ComputeTwoBodyHamiltonian");
  T2(a, b, i, j) = t2;
  Z2(a, b, i, j) = {0.0, 0.0};
  Z2(a, b, i, j) += H2(a, k, c, i) * T2(b, c, k, j);
  Z2(a, b, i, j) += H2(c, k, d, l) * T2(c, d, i, j) * T2(a, b, k, l);
  Z2(a, b, i, j) *= 0.25;
  T2(a, b, i, j) += Z2(a, b, i, j);
  talsh_t2 = T2;
  X2() = {0.0, 0.0};
  X2() += Z2 + (a, b, i, j) * Z2(a, b, i, j);
  norm_x2 = norm1(X2);
  talsh_x2 = X2;
  save X2 : tag("Z2_norm2");
  ~X2;
  ~Z2;
  destroy T2, H2;
  end scope main;
}

int main() {

  exatn::initialize();
  // Register a user-defined tensor method (simply performs random
  // initialization):
  exatn::registerTensorMethod(
      "ComputeTwoBodyHamiltonian",
      std::make_shared<exatn::numerics::FunctorInitRnd>());

  // Externally provided std::vector with user data used to init T2 (simply set
  // to a specific value):
  const std::size_t t2_tensor_volume = (84 - 42 + 1) * (84 - 42 + 1) *
                                       (127 - 0 + 1) *
                                       (127 - 0 + 1); 
  std::vector<std::complex<float>> data_vector(
      t2_tensor_volume, std::complex<float>{-1e-3, 1e-4});

  std::shared_ptr<talsh::Tensor> talsh_t2, talsh_x2;
  double norm_x2;
  test(data_vector, talsh_t2, talsh_x2, norm_x2);
}
