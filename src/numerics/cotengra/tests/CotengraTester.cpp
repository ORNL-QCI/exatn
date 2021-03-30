#include "contraction_seq_optimizer.hpp"
#include "exatn.hpp"
#include <gtest/gtest.h>

TEST(CotengraTester, checkSimple) {
  using exatn::Tensor;
  using exatn::TensorElementType;
  using exatn::TensorNetwork;
  using exatn::TensorShape;

  // exatn::resetLoggingLevel(1,2); //debug

  const unsigned int num_qubits = 53;
  std::vector<std::pair<unsigned int, unsigned int>> sycamore_8_cnot{
      {1, 4},   {3, 7},   {5, 9},   {6, 13},  {8, 15},  {10, 17}, {12, 21},
      {14, 23}, {16, 25}, {18, 27}, {20, 30}, {22, 32}, {24, 34}, {26, 36},
      {29, 37}, {31, 39}, {33, 41}, {35, 43}, {38, 44}, {40, 46}, {42, 48},
      {45, 49}, {47, 51}, {50, 52}, {0, 3},   {2, 6},   {4, 8},   {7, 14},
      {9, 16},  {11, 20}, {13, 22}, {15, 24}, {17, 26}, {19, 29}, {21, 31},
      {23, 33}, {25, 35}, {30, 38}, {32, 40}, {34, 42}, {39, 45}, {41, 47},
      {46, 50}};

  std::cout << "Building the circuit ... \n" << std::flush;

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

  std::cout << "HOWDY\n";
  circuit.printIt();
  auto cotengra =
      exatn::getService<exatn::numerics::ContractionSeqOptimizer>("cotengra");
  std::list<exatn::numerics::ContrTriple> results;
  cotengra->determineContractionSequence(
      circuit, results, [&]() -> unsigned int { return ++tensor_counter; });

  for (const auto &ctrTrip : results) {
    std::cout << "Contract: " << ctrTrip.left_id << " and " << ctrTrip.right_id << " --> "
              << ctrTrip.result_id << "\n";
  }
}

int main(int argc, char **argv) {
  exatn::initialize();

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();

  exatn::finalize();
  return ret;
}
