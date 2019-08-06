/** ExaTN: Numerics: Symbolic tensor processing
REVISION: 2019/08/06

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
  if(tensor[left_par_pos-1] == '+'){ //complex conjugation
   if(left_par_pos < 2) return false;
   complex_conjugated = true;
   tensor_name = tensor.substr(0,left_par_pos-1);
  }else{
   complex_conjugated = false;
   tensor_name = tensor.substr(0,left_par_pos);
  }
  //Extract tensor indices:
  if(right_par_pos > (left_par_pos + 1)){
   auto label_direction = LegDirection::UNDIRECT;
   if(tensor.find("|") != std::string::npos) label_direction = LegDirection::OUTWARD;
   auto i = left_par_pos + 1;
   auto label_begin = i;
   while(i <= right_par_pos){
    if(tensor[i] == ',' || tensor[i] == '|' || tensor[i] == ')'){ //index separator
     auto label_end = (i-1);
     if(label_end < label_begin) return false;
     indices.emplace_back(IndexLabel{tensor.substr(label_begin,label_end-label_begin+1),label_direction});
     label_begin = (i+1);
     if(tensor[i] == '|'){
      if(label_direction == LegDirection::OUTWARD){
       label_direction = LegDirection::INWARD;
      }else{
       return false;
      }
     }else if(tensor[i] == ')'){
      if(i != right_par_pos) return false;
     }
    }
    ++i;
   }
  }
 }else{
  return false;
 }
 return true;
}

} //namespace exatn
