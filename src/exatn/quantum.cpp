/** ExaTN: Quantum computing related
REVISION: 2021/06/28

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include <iostream>
#include <fstream>

#include "quantum.hpp"

namespace exatn{

// All quantum gates will look transposed because of the column-wise data storage:
const std::vector<std::complex<double>> GATE_I {{1.0, 0.0}, {0.0, 0.0},
                                                {0.0, 0.0}, {1.0, 0.0}};
const std::vector<std::complex<double>> GATE_X {{0.0, 0.0}, {1.0, 0.0},
                                                {1.0, 0.0}, {0.0, 0.0}};
const std::vector<std::complex<double>> GATE_Y {{0.0, 0.0}, {0.0, 1.0},
                                                {0.0,-1.0}, {0.0, 0.0}};
const std::vector<std::complex<double>> GATE_Z {{1.0, 0.0}, {0.0, 0.0},
                                                {0.0, 0.0}, {-1.0,0.0}};
const std::vector<std::complex<double>> GATE_H {{1.0, 0.0}, {1.0, 0.0},
                                                {1.0, 0.0}, {-1.0,0.0}};
const std::vector<std::complex<double>> GATE_S {{1.0, 0.0}, {0.0, 0.0},
                                                {0.0, 0.0}, {0.0, 1.0}};
const std::vector<std::complex<double>> GATE_T {{1.0, 0.0}, {0.0, 0.0},
                                                {0.0, 0.0}, {std::sqrt(2.0)*0.5, std::sqrt(2.0)*0.5}};


const std::vector<std::complex<double>> & getQuantumGateData(const QuantumGate gate_name,
                                                             std::initializer_list<double> angles)
{
 switch(gate_name){
  case(QuantumGate::gate_I): return GATE_I;
  case(QuantumGate::gate_X): return GATE_X;
  case(QuantumGate::gate_Y): return GATE_Y;
  case(QuantumGate::gate_Z): return GATE_Z;
  case(QuantumGate::gate_H): return GATE_H;
  case(QuantumGate::gate_S): return GATE_S;
  case(QuantumGate::gate_T): return GATE_T;
 }
 std::cout << "#ERROR(exatn::quantum): Unknown quantum gate!" << std::endl;
 assert(false);
}


bool appendPauliComponent(numerics::TensorOperator & tens_operator,
                          const std::string & paulis,
                          const std::complex<double> & coef)
{
 //`Finish: Append a Pauli strig component
 return true;
}

std::shared_ptr<numerics::TensorOperator> readSpinHamiltonian(const std::string & operator_name,
                                                              const std::string & filename,
                                                              unsigned int num_sites)
{
 assert(filename.length() > 0);
 assert(num_sites > 0);
 std::ifstream input_file(filename);
 auto tens_oper = makeSharedTensorOperator(operator_name);
 std::string line;
 while(std::getline(input_file,line)){
  std::string paulis;
  std::complex<double> coef;
  auto success = parse_pauli_string(line,paulis,coef); assert(success);
  success = appendPauliComponent(*tens_oper,paulis,coef); assert(success);
  line.clear();
 }
 input_file.close();
 return tens_oper;
}

} //namespace exatn
