/** ExaTN: Numerics: Symbolic tensor processing
REVISION: 2019/08/07

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_symbol.hpp"

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
  if(tensor[trim_range.second] == '+'){ //complex conjugation
   if(trim_range.second < trim_range.first + 1) return false;
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

} //namespace exatn
