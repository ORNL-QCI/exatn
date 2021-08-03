/** ExaTN: Quantum computing related
REVISION: 2021/08/03

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include <iostream>
#include <fstream>

#include "quantum.hpp"

namespace exatn{

namespace quantum{

/** All quantum gates will look transposed because of the column-wise data storage,
    that is, following the Big Endian convention, such that a qubit order reversal
    will be needed to match the standard text book (matrix) convention, for example:
    CX(i,j|k,l): {i,j} & {k,l} --> {00,10,01,11}, which requires q1q0 order instead
    of q0q1 in order to match the standard matrix-based gate definitions.
**/
const std::vector<std::complex<double>> GATE_0 {{0.0, 0.0}, {0.0, 0.0},
                                                {0.0, 0.0}, {0.0, 0.0}};
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
const std::vector<std::complex<double>> GATE_CX {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                                 {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                                 {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
                                                 {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
const std::vector<std::complex<double>> GATE_CY {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                                 {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                                 {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 1.0},
                                                 {0.0, 0.0}, {0.0, 0.0}, {0.0,-1.0}, {0.0, 0.0}};
const std::vector<std::complex<double>> GATE_CZ {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                                 {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                                 {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
                                                 {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0,0.0}};
const std::vector<std::complex<double>> GATE_SWAP {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                                   {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
                                                   {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                                   {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
const std::vector<std::complex<double>> GATE_ISWAP {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                                    {0.0, 0.0}, {0.0, 0.0}, {0.0, 1.0}, {0.0, 0.0},
                                                    {0.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
                                                    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};


std::vector<std::complex<double>> getGateData(const Gate gate_name,
                                              std::initializer_list<double> angles)
{
 std::vector<std::complex<double>> gate_data;
 switch(gate_name){
  case(Gate::gate_0): gate_data = GATE_0; break;
  case(Gate::gate_I): gate_data = GATE_I; break;
  case(Gate::gate_X): gate_data = GATE_X; break;
  case(Gate::gate_Y): gate_data = GATE_Y; break;
  case(Gate::gate_Z): gate_data = GATE_Z; break;
  case(Gate::gate_H): gate_data = GATE_H; break;
  case(Gate::gate_S): gate_data = GATE_S; break;
  case(Gate::gate_T): gate_data = GATE_T; break;
  case(Gate::gate_CX): gate_data = GATE_CX; break;
  case(Gate::gate_CY): gate_data = GATE_CY; break;
  case(Gate::gate_CZ): gate_data = GATE_CZ; break;
  case(Gate::gate_SWAP): gate_data = GATE_SWAP; break;
  case(Gate::gate_ISWAP): gate_data = GATE_ISWAP; break;
  default:
   std::cout << "#ERROR(exatn::quantum::getGateData): Unknown quantum gate!" << std::endl;
   assert(false);
 }
 return gate_data;
}


bool appendPauliComponent(exatn::numerics::TensorOperator & tens_operator,
                          const std::string & paulis,
                          const std::complex<double> & coef,
                          TensorElementType precision)
{
 bool success = true;
 std::shared_ptr<exatn::numerics::Tensor> gate_tensor;
 auto pauli_product = std::make_shared<exatn::numerics::TensorNetwork>();
 std::vector<std::pair<unsigned int, unsigned int>> ket_pairing;
 std::vector<std::pair<unsigned int, unsigned int>> bra_pairing;
 unsigned int pauli_id = 0;
 std::size_t pos = 1;
 while(paulis[pos] != ']'){
  auto gate_name = Gate::gate_I;
  if(paulis[pos] == 'I'){
   gate_name = Gate::gate_I;
   if(!exatn::tensorAllocated("_Pauli_I")){
    success = exatn::createTensorSync("_Pauli_I",precision,TensorShape{2,2});
    if(success) success = exatn::initTensorDataSync("_Pauli_I",getGateData(gate_name));
   }
   gate_tensor = exatn::getTensor("_Pauli_I");
  }else if(paulis[pos] == 'X'){
   gate_name = Gate::gate_X;
   if(!exatn::tensorAllocated("_Pauli_X")){
    success = exatn::createTensorSync("_Pauli_X",precision,TensorShape{2,2});
    if(success) success = exatn::initTensorDataSync("_Pauli_X",getGateData(gate_name));
   }
   gate_tensor = exatn::getTensor("_Pauli_X");
  }else if(paulis[pos] == 'Y'){
   gate_name = Gate::gate_Y;
   if(!exatn::tensorAllocated("_Pauli_Y")){
    success = exatn::createTensorSync("_Pauli_Y",precision,TensorShape{2,2});
    if(success) success = exatn::initTensorDataSync("_Pauli_Y",getGateData(gate_name));
   }
   gate_tensor = exatn::getTensor("_Pauli_Y");
  }else if(paulis[pos] == 'Z'){
   gate_name = Gate::gate_Z;
   if(!exatn::tensorAllocated("_Pauli_Z")){
    success = exatn::createTensorSync("_Pauli_Z",precision,TensorShape{2,2});
    if(success) success = exatn::initTensorDataSync("_Pauli_Z",getGateData(gate_name));
   }
   gate_tensor = exatn::getTensor("_Pauli_Z");
  }else{
   std::cout << "#ERROR(exatn::quantum::readSpinHamiltonian): Invalid Pauli gate: " << paulis << std::endl;
   success = false;
  }
  success = success && (gate_tensor != nullptr);
  if(!success) break;
  ++pos;
  const auto beg_pos = pos;
  while(is_number(paulis[pos])) ++pos;
  unsigned int qubit = static_cast<unsigned int>(std::stoi(paulis.substr(beg_pos,pos-beg_pos)));
  while(paulis[pos] == ' ') ++pos;
  success = pauli_product->appendTensor(gate_tensor,{}); if(!success) break;
  ket_pairing.push_back({qubit,pauli_id*2+1});
  bra_pairing.push_back({qubit,pauli_id*2+0});
  ++pauli_id;
 }
 pauli_product->rename("PauliProduct" + std::to_string(tens_operator.getNumComponents()));
 success = tens_operator.appendComponent(pauli_product,ket_pairing,bra_pairing,coef);
 return success;
}


std::shared_ptr<exatn::numerics::TensorOperator> readSpinHamiltonian(const std::string & operator_name,
                                                                     const std::string & filename,
                                                                     TensorElementType precision)
{
 assert(filename.length() > 0);
 assert(precision == TensorElementType::COMPLEX32 || precision == TensorElementType::COMPLEX64);
 std::ifstream input_file(filename);
 if(!input_file){
  std::cout << "#ERROR(exatn::quantum::readSpinHamiltonian): File not found: " << filename << std::endl;
  assert(false);
 }
 auto tens_oper = makeSharedTensorOperator(operator_name);
 std::string line;
 while(std::getline(input_file,line)){
  std::string paulis;
  std::complex<double> coef;
  auto success = parse_pauli_string(line,paulis,coef); assert(success);
  //std::cout << "#DEBUG: " << paulis << std::endl; //debug
  assert(paulis.length() >= 2); //'[]' at least
  assert(paulis[0] == '[' && paulis[paulis.length()-1] == ']');
  success = appendPauliComponent(*tens_oper,paulis,coef,precision); assert(success);
  line.clear();
 }
 input_file.close();
 return tens_oper;
}

} //namespace quantum

} //namespace exatn
