/** ExaTN: Quantum computing related
REVISION: 2021/11/02

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include <iostream>
#include <fstream>
#include <cmath>

#include "quantum.hpp"

namespace exatn{

namespace quantum{

//Constant gates:
const std::vector<std::complex<double>> GATE_0 {{0.0, 0.0}, {0.0, 0.0},
                                                {0.0, 0.0}, {0.0, 0.0}};
const std::vector<std::complex<double>> GATE_I {{1.0, 0.0}, {0.0, 0.0},
                                                {0.0, 0.0}, {1.0, 0.0}};
const std::vector<std::complex<double>> GATE_X {{0.0, 0.0}, {1.0, 0.0},
                                                {1.0, 0.0}, {0.0, 0.0}};
const std::vector<std::complex<double>> GATE_Y {{0.0, 0.0}, {0.0,-1.0},
                                                {0.0, 1.0}, {0.0, 0.0}};
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
                                                 {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0,-1.0},
                                                 {0.0, 0.0}, {0.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}};
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

//Single-parameter gates:
auto GATE_RX = [] (double theta) -> const std::vector<std::complex<double>> {
 auto th = theta * 0.5;
 return std::vector<std::complex<double>> {{std::cos(th),0.0},  {0.0,-std::sin(th)},
                                           {0.0,-std::sin(th)}, {std::cos(th),0.0}};
};

auto GATE_RY = [] (double theta) -> const std::vector<std::complex<double>> {
 auto th = theta * 0.5;
 return std::vector<std::complex<double>> {{std::cos(th),0.0}, {-std::sin(th),0.0},
                                           {std::sin(th),0.0}, {std::cos(th),0.0}};
};

auto GATE_RZ = [] (double theta) -> const std::vector<std::complex<double>> {
 auto th = theta * 0.5;
 return std::vector<std::complex<double>> {{std::cos(th),-std::sin(th)}, {0.0,0.0},
                                           {0.0,0.0}, {std::cos(th),std::sin(th)}};
};

auto GATE_CR = [] (double theta) -> const std::vector<std::complex<double>> {
 return std::vector<std::complex<double>> {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                           {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                           {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
                                           {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {std::cos(theta),std::sin(theta)}};
};


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
  case(Gate::gate_Rx):
   assert(angles.size() == 1);
   gate_data = GATE_RX(*(angles.begin()));
   break;
  case(Gate::gate_Ry):
   assert(angles.size() == 1);
   gate_data = GATE_RY(*(angles.begin()));
   break;
  case(Gate::gate_Rz):
   assert(angles.size() == 1);
   gate_data = GATE_RZ(*(angles.begin()));
   break;
  case(Gate::gate_CX): gate_data = GATE_CX; break;
  case(Gate::gate_CY): gate_data = GATE_CY; break;
  case(Gate::gate_CZ): gate_data = GATE_CZ; break;
  case(Gate::gate_SWAP): gate_data = GATE_SWAP; break;
  case(Gate::gate_ISWAP): gate_data = GATE_ISWAP; break;
  case(Gate::gate_CR):
   assert(angles.size() == 1);
   gate_data = GATE_CR(*(angles.begin()));
   break;
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
                                                                     TensorElementType precision,
                                                                     const std::string & format)
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
  auto success = false;
  if(format == "OpenFermion"){
   success = parse_pauli_string_ofermion(line,paulis,coef);
  }else if(format == "QCWare"){
   success = parse_pauli_string_qcware(line,paulis,coef);
  }
  if(!success){
   std::cout << "#ERROR(exatn:quantum:readSpinHamiltonian): Unable to parse file "
             << filename << " with format " << format << std::endl;
   assert(false);
  }
  //std::cout << "#DEBUG: " << paulis << std::endl; //debug
  assert(paulis.length() >= 2); //'[]' at least
  assert(paulis[0] == '[' && paulis[paulis.length()-1] == ']');
  success = appendPauliComponent(*tens_oper,paulis,coef,precision); assert(success);
  line.clear();
 }
 input_file.close();
 return tens_oper;
}


std::shared_ptr<exatn::numerics::TensorOperator> generateSpinHamiltonian(const std::string & operator_name,
                                                  std::function<PauliProduct ()> hamiltonian_generator,
                                                  TensorElementType precision)
{
 auto hamiltonian = exatn::makeSharedTensorOperator(operator_name);
 while(true){
  const auto pauli_prod = hamiltonian_generator();
  if(pauli_prod.product.size() == 0 && pauli_prod.coefficient == std::complex<double>{0.0,0.0}) break;
  std::string paulis = "[";
  bool not_first = false;
  for(const auto & pauli: pauli_prod.product){
   if(not_first) paulis += " ";
   if(pauli.pauli_gate == Gate::gate_I){
    paulis += "I";
   }else if(pauli.pauli_gate == Gate::gate_X){
    paulis += "X";
   }else if(pauli.pauli_gate == Gate::gate_Y){
    paulis += "Y";
   }else if(pauli.pauli_gate == Gate::gate_Z){
    paulis += "Z";
   }else{
    std::cout << "#ERROR(exatn::quantum::generateSpinHamiltonian): Invalid gate returned by the generator: "
              << static_cast<int>(pauli.pauli_gate) << std::endl;
    assert(false);
   }
   paulis += std::to_string(pauli.qubit);
   not_first = true;
  }
  paulis += "]";
  auto success = appendPauliComponent(*hamiltonian,paulis,pauli_prod.coefficient,precision); assert(success);
 }
 return hamiltonian;
}

} //namespace quantum

} //namespace exatn
