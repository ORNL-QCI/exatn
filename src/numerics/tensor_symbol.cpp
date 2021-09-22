/** ExaTN: Numerics: Symbolic tensor processing
REVISION: 2021/09/22

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include <iostream>

#include "tensor_symbol.hpp"

#include <map>

namespace exatn{

bool parse_tensor(const std::string & tensor,        //in: tensor as a string
                  std::string & tensor_name,         //out: tensor name
                  std::vector<IndexLabel> & indices, //out: tensor indices (labels)
                  bool & complex_conjugated)         //out: whether or not tensor appears complex conjugated
{
 //Check function arguments:
 if(tensor.empty()) return false;
 //Clear output function arguments:
 tensor_name.clear();
 indices.clear();
 //Find left/right parentheses:
 auto left_par_pos = tensor.find("(");
 if(left_par_pos == std::string::npos) return false;
 auto right_par_pos = tensor.rfind(")");
 if(right_par_pos == std::string::npos) return false;
 if(left_par_pos >= right_par_pos) return false;
 if(left_par_pos > 0){ //tensor name must be non-empty
  //Check for tensor complex conjugation:
  auto trim_range = trim_spaces_off(tensor,std::pair<int,int>{0,static_cast<int>(left_par_pos-1)});
  if(trim_range.first > trim_range.second) return false;
  if(tensor[trim_range.second] == '+'){ //complex conjugation sign
   if(trim_range.second < trim_range.first + 1) return false;
   --trim_range.second;
   complex_conjugated = true;
  }else{
   complex_conjugated = false;
  }
  tensor_name = tensor.substr(trim_range.first,trim_range.second-trim_range.first+1);
  if(!is_alphanumeric(tensor_name)) return false;
  //Extract tensor indices:
  if(right_par_pos > (left_par_pos + 1)){ //indices present
   auto label_direction = LegDirection::UNDIRECT;
   if(tensor.find("|") != std::string::npos) label_direction = LegDirection::OUTWARD;
   auto i = left_par_pos + 1;
   auto label_begin = i;
   auto label_end = label_begin - 1;
   bool label_started = false;
   bool label_finished = false;
   while(i <= right_par_pos){
    if(tensor[i] == ' '){
     if(label_started) label_finished = true;
    }else if(tensor[i] == ',' || tensor[i] == '|' || tensor[i] == ')'){ //index separator
     if(label_started){
      label_finished = true;
     }else{
      return false;
     }
     if(label_end < label_begin) return false;
     trim_range = trim_spaces_off(tensor,std::pair<int,int>{static_cast<int>(label_begin),static_cast<int>(label_end)});
     indices.emplace_back(IndexLabel{tensor.substr(trim_range.first,trim_range.second-trim_range.first+1),label_direction});
     if(!is_alphanumeric(indices.back().label)) return false;
     if(tensor[i] == '|'){
      if(label_direction == LegDirection::OUTWARD){
       label_direction = LegDirection::INWARD;
      }else{
       return false;
      }
     }else if(tensor[i] == ')'){
      if(i != right_par_pos) return false;
     }
     label_started = false;
     label_finished = false;
    }else{
     if(label_finished) return false;
     if(!label_started){
      label_started = true;
      label_begin = i;
     }
     label_end = i;
    }
    ++i;
   }
  }
 }else{
  return false;
 }
 return true;
}


bool parse_tensor_network(const std::string & network,        //in: tensor network as a string
                          std::vector<std::string> & tensors) //out: parsed (symbolic) tensors
{
 if(network.empty()) return false;
 tensors.clear();
 //Find the output tensor (l.h.s. tensor):
 auto net_len = network.length();
 auto equal_pos = network.find("=");
 if(equal_pos == std::string::npos) return false;
 if(equal_pos < 3) return false; //l.h.s. (output) tensor consists of at least three characters
 //Extract the output tensor:
 auto output_end = equal_pos - 1;
 if(network[output_end] == '+') --output_end; //"+=" equality sign use instead of "="
 auto trim_range = trim_spaces_off(network,std::pair<int,int>{0,static_cast<int>(output_end)});
 if(trim_range.first > trim_range.second) return false;
 tensors.emplace_back(network.substr(trim_range.first,trim_range.second-trim_range.first+1));
 //Extract the input tensors:
 int input_beg = equal_pos + 1;
 bool input_tensor_found = true;
 while(input_tensor_found){
  auto mult_pos = network.find("*",input_beg);
  input_tensor_found = (mult_pos != std::string::npos);
  if(input_tensor_found){
   int input_end = mult_pos - 1;
   trim_range = trim_spaces_off(network,std::pair<int,int>{input_beg,input_end});
   if(trim_range.first > trim_range.second) return false;
   tensors.emplace_back(network.substr(trim_range.first,trim_range.second-trim_range.first+1));
   input_beg = mult_pos + 1;
  }
 }
 //Extract the last tensor:
 if(input_beg < net_len){
  trim_range = trim_spaces_off(network,std::pair<int,int>{input_beg,static_cast<int>(net_len-1)});
  if(trim_range.first > trim_range.second) return false;
  tensors.emplace_back(network.substr(trim_range.first,trim_range.second-trim_range.first+1));
 }else{
  return false;
 }
 return true;
}


bool parse_tensor_contraction(const std::string & contraction,            //in: binary tensor contraction as a string
                              std::vector<std::string> & tensors,         //out: parsed (symbolic) tensors
                              std::vector<PosIndexLabel> & left_indices,  //out: left indices
                              std::vector<PosIndexLabel> & right_indices, //out: right indices
                              std::vector<PosIndexLabel> & contr_indices, //out: contracted indices
                              std::vector<PosIndexLabel> & hyper_indices) //out: hyper indices
{
 assert(contraction.length() > 0);
 bool success = parse_tensor_network(contraction,tensors);
 if(success){
  assert(tensors.size() == 3); //three tensor operands

  typedef struct{int arg_id[3];} IndexPositions;
  const IndexPositions void_pos{{-1,-1,-1}};

  auto compare = [](const IndexLabel & item1, const IndexLabel & item2){
   return item1.label < item2.label;
  };

  std::map<IndexLabel,IndexPositions,decltype(compare)> labels(compare);
  for(unsigned int i = 0; i < 3; ++i){
   std::vector<IndexLabel> indices;
   std::string tensor_name;
   bool conj;
   success = parse_tensor(tensors[i],tensor_name,indices,conj);
   if(success){
    for(int pos = 0; pos < indices.size(); ++pos){
     auto res = labels.emplace(std::make_pair(indices[pos],void_pos));
     assert(res.first->second.arg_id[i] < 0);
     res.first->second.arg_id[i] = pos;
    }
   }else{
    break;
   }
  }
  if(success){
   left_indices.clear();
   right_indices.clear();
   contr_indices.clear();
   hyper_indices.clear();
   for(const auto & kv: labels){
    PosIndexLabel ind_label{kv.first,{-1,-1,-1},0};
    for(int j = 0; j < 3; ++j) ind_label.arg_pos[j] = kv.second.arg_id[j];
    auto ind_kind = indexKind(ind_label);
    switch(ind_kind){
    case IndexKind::LEFT:
     left_indices.emplace_back(ind_label);
     break;
    case IndexKind::RIGHT:
     right_indices.emplace_back(ind_label);
     break;
    case IndexKind::CONTR:
     contr_indices.emplace_back(ind_label);
     break;
    case IndexKind::HYPER:
     hyper_indices.emplace_back(ind_label);
     break;
    default:
     std::cout << "#ERROR(parse_tensor_contraction): Invalid index kind!" << std::endl << std::flush;
     assert(false);
    }
   }
  }
 }
 return success;
}


std::string assemble_symbolic_tensor(const std::string & tensor_name,         //in: tensor name
                                     const std::vector<IndexLabel> & indices, //in: tensor indices
                                     bool conjugated)
{
 assert(tensor_name.length() > 0);
 std::string tensor(tensor_name);
 if(!conjugated){
  tensor += "(";
 }else{
  tensor += "+(";
 }
 for(const auto & index: indices) tensor += (index.label + ",");
 if(tensor[tensor.length()-1] == ','){
  tensor[tensor.length()-1] = ')';
 }else{
  tensor += ")";
 }
 return tensor;
}


std::string assemble_symbolic_tensor_network(const std::vector<std::string> & tensors)
{
 std::string tensor_network;
 const unsigned int num_tensors = tensors.size();
 assert(num_tensors >= 2);
 tensor_network = (tensors[0] + "+=" + tensors[1]);
 for(unsigned int i = 2; i < num_tensors; ++i) tensor_network += ("*" + tensors[i]);
 return std::move(tensor_network);
}


bool generate_contraction_pattern(const std::vector<numerics::TensorLeg> & pattern,
                                  unsigned int left_tensor_rank,
                                  unsigned int right_tensor_rank,
                                  std::string & symb_pattern,
                                  bool left_conjugated,
                                  bool right_conjugated,
                                  const std::string & dest_name,
                                  const std::string & left_name,
                                  const std::string & right_name)
/* pattern[left_rank + right_rank] = {left_legs + right_legs} */
{
 const std::size_t DEFAULT_STRING_CAPACITY = 256; //string capacity reserve value

 assert(pattern.size() == left_tensor_rank + right_tensor_rank);
 symb_pattern.clear();
 if(pattern.empty()){ //multiplication of scalars
  if(left_conjugated && right_conjugated){
   symb_pattern = dest_name+"()+="+left_name+"+()*"+right_name+"+()";
  }else if(left_conjugated && !right_conjugated){
   symb_pattern = dest_name+"()+="+left_name+"+()*"+right_name+"()";
  }else if(!left_conjugated && right_conjugated){
   symb_pattern = dest_name+"()+="+left_name+"()*"+right_name+"+()";
  }else{
   symb_pattern = dest_name+"()+="+left_name+"()*"+right_name+"()";
  }
 }else{ //at least one tensor is present
  if(symb_pattern.capacity() < DEFAULT_STRING_CAPACITY) symb_pattern.reserve(DEFAULT_STRING_CAPACITY);
  unsigned int dest_indices[left_tensor_rank + right_tensor_rank];
  unsigned int dest_tensor_rank = 0;
  for(const auto & leg: pattern){
   if(leg.getTensorId() == 0){
    dest_indices[leg.getDimensionId()] = dest_tensor_rank++;
   }
  }
  symb_pattern.append(dest_name+"(");
  for(unsigned int i = 0; i < dest_tensor_rank; ++i){
   symb_pattern.append("u"+std::to_string(dest_indices[i])+",");
  }
  if(symb_pattern[symb_pattern.size()-1] == ','){
   symb_pattern.replace(symb_pattern.size()-1,1,")");
  }else{
   symb_pattern.append(")");
  }
  if(left_conjugated){
   symb_pattern.append("+="+left_name+"+(");
  }else{
   symb_pattern.append("+="+left_name+"(");
  }
  dest_tensor_rank = 0;
  unsigned int contr_ind = 0;
  for(unsigned int i = 0; i < left_tensor_rank; ++i){
   if(pattern[i].getTensorId() == 0){
    dest_indices[i] = left_tensor_rank;
    symb_pattern.append("u"+std::to_string(dest_tensor_rank++)+",");
   }else{
    dest_indices[i] = contr_ind;
    symb_pattern.append("c"+std::to_string(contr_ind++)+",");
   }
  }
  if(symb_pattern[symb_pattern.size()-1] == ','){
   symb_pattern.replace(symb_pattern.size()-1,1,")");
  }else{
   symb_pattern.append(")");
  }
  if(right_conjugated){
   symb_pattern.append("*"+right_name+"+(");
  }else{
   symb_pattern.append("*"+right_name+"(");
  }
  for(unsigned int i = left_tensor_rank; i < left_tensor_rank + right_tensor_rank; ++i){
   if(pattern[i].getTensorId() == 0){
    symb_pattern.append("u"+std::to_string(dest_tensor_rank++)+",");
   }else{
    contr_ind = dest_indices[pattern[i].getDimensionId()];
    assert(contr_ind < left_tensor_rank);
    symb_pattern.append("c"+std::to_string(contr_ind)+",");
   }
  }
  if(symb_pattern[symb_pattern.size()-1] == ','){
   symb_pattern.replace(symb_pattern.size()-1,1,")");
  }else{
   symb_pattern.append(")");
  }
 }
 /*{//DEBUG:
  std::cout << std::endl;
  for(const auto & leg: pattern) leg.printIt();
  std::cout << " " << symb_pattern << std::endl;
 }*/
 return true;
}


bool generate_addition_pattern(const std::vector<numerics::TensorLeg> & pattern,
                               std::string & symb_pattern,
                               bool conjugated,
                               const std::string & dest_name,
                               const std::string & left_name)
/* pattern[left_rank] = {left_legs} */
{
 unsigned int rank = pattern.size();
 auto generated = generate_contraction_pattern(pattern,rank,0,symb_pattern,
                                               conjugated,false,dest_name,left_name);
 if(generated){
  auto pos = symb_pattern.rfind("*R()");
  generated = (pos != std::string::npos);
  if(generated) symb_pattern.erase(pos);
 }
 //if(generated) std::cout << symb_pattern << std::endl; //debug
 return generated;
}


/* Generates the trivial tensor addition pattern. */
bool generate_addition_pattern(unsigned int tensor_rank,
                               std::string & symb_pattern,
                               bool conjugated,
                               const std::string & dest_name,
                               const std::string & left_name)
{
 std::vector<numerics::TensorLeg> pattern(tensor_rank);
 unsigned int dim = 0;
 for(auto & leg: pattern) leg = numerics::TensorLeg(0,dim++);
 return generate_addition_pattern(pattern,symb_pattern,conjugated,dest_name,left_name);
}

bool parse_pauli_string(const std::string & input,
                        std::string & paulis,
                        std::complex<double> & coefficient)
{
 bool success = true;
 double coef_real, coef_imag;
 const auto left_par_pos = input.find("(");
 if(left_par_pos != std::string::npos){
  const auto right_par_pos = input.find(")",left_par_pos);
  if(right_par_pos != std::string::npos){
   const auto left_sq_pos = input.find("[",right_par_pos);
   if(left_sq_pos != std::string::npos){
    const auto right_sq_pos = input.find("]",left_sq_pos);
    if(right_sq_pos != std::string::npos){
     paulis = input.substr(left_sq_pos,right_sq_pos-left_sq_pos+1);
     const auto plus_pos = input.find_last_of("+",right_par_pos);
     const auto minus_pos = input.find_last_of("-",right_par_pos);
     auto sep_pos = plus_pos;
     if(minus_pos == std::string::npos){
      sep_pos = plus_pos;
     }else{
      if(plus_pos == std::string::npos){
       sep_pos = minus_pos;
      }else{
       sep_pos = std::max(minus_pos,plus_pos);
      }
     }
     if(sep_pos != std::string::npos){
      const auto real_len = sep_pos - left_par_pos - 1;
      //std::cout << "#DEBUG(parse_pauli_string): Coef: " << input.substr(left_par_pos+1,real_len); //debug
      if(real_len > 0) coef_real = std::stod(input.substr(left_par_pos+1,real_len));
      const auto imag_end_pos = input.find("j",sep_pos);
      if(imag_end_pos != std::string::npos){
       const auto imag_len = imag_end_pos - sep_pos - 1;
       //std::cout << " " << input.substr(sep_pos+1,imag_len) << std::endl; //debug
       if(imag_len > 0) coef_imag = std::stod(input.substr(sep_pos+1,imag_len));
       coefficient = std::complex<double>{coef_real, coef_imag};
      }else{
       success = false;
      }
     }else{
      success = false;
     }
    }else{
     success = false;
    }
   }else{
    success = false;
   }
  }else{
   success = false;
  }
 }else{
  success = false;
 }
 return success;
}

} //namespace exatn
