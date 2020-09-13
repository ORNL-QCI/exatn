#include <vector>
#include <complex>

[[clang::syntax(taprol)]] void test(std::vector<std::complex<double>> & t2_data,
                                    std::shared_ptr<talsh::Tensor> talsh_t2,
                                    std::shared_ptr<talsh::Tensor> talsh_x2,
                                    double & norm_x2) {
  entry: main;

  scope main group();
    space(complex) : space0 = [0:255], space1 = [0:511];
    subspace(space0) : s0 = [0:127], s1 = [128:255];
    subspace(space1) : r0 = [0:283], r1 = [284:511];
    index(s0) : i, j, k, l;
    index(r0) : a, b, c, d;
    H2(a, i, b, j) = {0.0, 0.0};
    H2(a, i, b, j) = method("ComputeTwoBodyHamiltonian");
    T2(a, b, i, j) = t2_data;
    Z2(a, b, i, j) = {0.0, 0.0};
    Z2(a, b, i, j) += H2(a, k, c, i) * T2(b, c, k, j);
    Z2(a, b, i, j) += H2(c, k, d, l) * T2(c, d, i, j) * T2(a, b, k, l);
    Z2(a, b, i, j) *= 0.25;
    T2(a, b, i, j) += Z2(a, b, i, j);
    talsh_t2 = T2;
    X2() = {0.0, 0.0};
    X2() += Z2+(a, b, i, j) * Z2(a, b, i, j);
    norm_x2 = norm1(X2);
    talsh_x2 = X2;
    save X2 : tag("Z2_norm1");
    ~X2;
    ~Z2;
    destroy T2, H2;
  end scope main;
}

int main() {

}
