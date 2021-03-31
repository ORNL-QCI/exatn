#include "contraction_seq_optimizer.hpp"
#include "exatn.hpp"
#include "talshxx.hpp"
#include <gtest/gtest.h>

TEST(CotengraTester, checkContractPath) {
  using exatn::Tensor;
  using exatn::TensorElementType;
  using exatn::TensorNetwork;
  using exatn::TensorShape;
  const unsigned int num_qubits = 53;
  std::vector<std::pair<unsigned int, unsigned int>> sycamore_8_cnot{
      {1, 4},   {3, 7},   {5, 9},   {6, 13},  {8, 15},  {10, 17}, {12, 21},
      {14, 23}, {16, 25}, {18, 27}, {20, 30}, {22, 32}, {24, 34}, {26, 36},
      {29, 37}, {31, 39}, {33, 41}, {35, 43}, {38, 44}, {40, 46}, {42, 48},
      {45, 49}, {47, 51}, {50, 52}, {0, 3},   {2, 6},   {4, 8},   {7, 14},
      {9, 16},  {11, 20}, {13, 22}, {15, 24}, {17, 26}, {19, 29}, {21, 31},
      {23, 33}, {25, 35}, {30, 38}, {32, 40}, {34, 42}, {39, 45}, {41, 47},
      {46, 50}};

  std::cout << "Building the circuit ... \n";
  TensorNetwork circuit("Sycamore8_CNOT");
  unsigned int tensor_counter = 0;

  // Left qubit tensors:
  unsigned int first_q_tensor = tensor_counter + 1;
  for (unsigned int i = 0; i < num_qubits; ++i) {
    bool success = circuit.appendTensor(
        ++tensor_counter,
        std::make_shared<Tensor>("Q" + std::to_string(i), TensorShape{2}), {});
    assert(success);
  }
  unsigned int last_q_tensor = tensor_counter;

  // CNOT gates:
  auto cnot = std::make_shared<Tensor>("CNOT", TensorShape{2, 2, 2, 2});
  for (unsigned int i = 0; i < sycamore_8_cnot.size(); ++i) {
    bool success = circuit.appendTensorGate(
        ++tensor_counter, cnot,
        {sycamore_8_cnot[i].first, sycamore_8_cnot[i].second});
    assert(success);
  }

  // Right qubit tensors:
  unsigned int first_p_tensor = tensor_counter + 1;
  for (unsigned int i = 0; i < num_qubits; ++i) {
    bool success = circuit.appendTensor(
        ++tensor_counter,
        std::make_shared<Tensor>("P" + std::to_string(i), TensorShape{2}),
        {{0, 0}});
    assert(success);
  }

  circuit.printIt();
  auto cotengra =
      exatn::getService<exatn::numerics::ContractionSeqOptimizer>("cotengra");
  std::list<exatn::numerics::ContrTriple> results;
  cotengra->determineContractionSequence(
      circuit, results, [&]() -> unsigned int { return ++tensor_counter; });

  for (const auto &ctrTrip : results) {
    std::cout << "Contract: " << ctrTrip.left_id << " and " << ctrTrip.right_id
              << " --> " << ctrTrip.result_id << "\n";
  }
}

TEST(CotengraTester, checkEvaluate) {
  using exatn::Tensor;
  using exatn::TensorElementType;
  using exatn::TensorNetwork;
  using exatn::TensorShape;
  // Use cotengra
  exatn::resetContrSeqOptimizer("cotengra");
  // Quantum Circuit:
  // Q0----H---------
  // Q1----H----C----
  // Q2----H----N----

  // Define the initial qubit state vector:
  std::vector<std::complex<double>> qzero{{1.0, 0.0}, {0.0, 0.0}};

  // Define quantum gates:
  std::vector<std::complex<double>> hadamard{
      {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {-1.0, 0.0}};
  std::vector<std::complex<double>> identity{
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
  std::vector<std::complex<double>> cnot{
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
      {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
      {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

  // Create qubit tensors:
  auto created = false;
  created =
      exatn::createTensor("Q0", TensorElementType::COMPLEX64, TensorShape{2});
  assert(created);
  created =
      exatn::createTensor("Q1", TensorElementType::COMPLEX64, TensorShape{2});
  assert(created);
  created =
      exatn::createTensor("Q2", TensorElementType::COMPLEX64, TensorShape{2});
  assert(created);

  // Create gate tensors:
  auto registered = false;
  created =
      exatn::createTensor("H", TensorElementType::COMPLEX64, TensorShape{2, 2});
  assert(created);
  registered = exatn::registerTensorIsometry("H", {0}, {1});
  assert(registered);
  created =
      exatn::createTensor("I", TensorElementType::COMPLEX64, TensorShape{2, 2});
  assert(created);
  registered = exatn::registerTensorIsometry("I", {0}, {1});
  assert(registered);
  created = exatn::createTensor("CNOT", TensorElementType::COMPLEX64,
                                TensorShape{2, 2, 2, 2});
  assert(created);
  registered = exatn::registerTensorIsometry("CNOT", {0, 1}, {2, 3});
  assert(registered);

  // Initialize qubit tensors to zero state:
  auto initialized = false;
  initialized = exatn::initTensorData("Q0", qzero);
  assert(initialized);
  initialized = exatn::initTensorData("Q1", qzero);
  assert(initialized);
  initialized = exatn::initTensorData("Q2", qzero);
  assert(initialized);

  // Initialize necessary gate tensors:
  initialized = exatn::initTensorData("H", hadamard);
  assert(initialized);
  initialized = exatn::initTensorData("CNOT", cnot);
  assert(initialized);
  initialized = exatn::initTensorData("I", identity);
  assert(initialized);

  { // Open a new scope:
    // Build a tensor network from the quantum circuit:
    TensorNetwork circuit("QuantumCircuit");
    auto appended = false;
    appended = circuit.appendTensor(1, exatn::getTensor("Q0"), {});
    assert(appended);
    appended = circuit.appendTensor(2, exatn::getTensor("Q1"), {});
    assert(appended);
    appended = circuit.appendTensor(3, exatn::getTensor("Q2"), {});
    assert(appended);

    appended = circuit.appendTensorGate(4, exatn::getTensor("H"), {0});
    assert(appended);
    appended = circuit.appendTensorGate(5, exatn::getTensor("CNOT"), {1, 0});
    assert(appended);
    appended = circuit.appendTensorGate(6, exatn::getTensor("CNOT"), {2, 0});
    assert(appended);
    appended = circuit.appendTensorGate(7, exatn::getTensor("I"), {0});
    assert(appended);
    appended = circuit.appendTensorGate(8, exatn::getTensor("I"), {1});
    assert(appended);
    appended = circuit.appendTensorGate(9, exatn::getTensor("I"), {2});
    assert(appended);
    circuit.printIt(); // debug

    // Evaluate the quantum circuit expressed as a tensor network:
    auto evaluated = false;
    evaluated = exatn::evaluateSync(circuit);
    assert(evaluated);
    auto local_copy = exatn::getLocalTensor(circuit.getTensor(0)->getName());
    assert(local_copy);
    const exatn::TensorDataType<TensorElementType::COMPLEX64>::value *body_ptr;
    auto access_granted = local_copy->getDataAccessHostConst(&body_ptr);
    assert(access_granted);

    const auto tensorVolume = local_copy->getVolume();
    assert(tensorVolume == 8);
    std::vector<std::complex<double>> waveFn;
    waveFn.assign(body_ptr, body_ptr + tensorVolume);
    body_ptr = nullptr;
    for (const auto &el : waveFn) {
      std::cout << el << "\n";
    }
    // Synchronize:
    exatn::sync();
  }
}

int main(int argc, char **argv) {
  exatn::initialize();

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();

  exatn::finalize();
  return ret;
}
