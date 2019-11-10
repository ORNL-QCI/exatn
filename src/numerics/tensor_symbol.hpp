/** ExaTN: Numerics: Symbolic tensor processing
REVISION: 2019/11/10

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
(a) Valid symbolic tensor formats are (angle brackets mark placeholders for tokens):
    (1) Scalar (rank-0 tensor):
        <TensorName><+>()
        Examples: S(), T+()
    (2) Tensor in an orthogonal basis:
        <TensorName><+>(<label>,<label>,...)
        Examples: Q(i1,a1), H+(p1,p2,q1,q2)
    (3) <TensorName><+>(<label>,<label>,...|<label>,<label>,...)
        Examples: Q(i1|a1), H+(p1,p2|q1,q2), R12(|a1,i3), L21(i1,b2|)
    where
     <TensorName> is an alphanumeric_ tensor name beginning with a letter;
     <+> is an optional complex conjugation sign;
     <label> is an alphanumeric_ index label beginning with a letter;
    In case the "|" separator is absent, all tensor indices are considered
    invariant, corresponding to undirected tensor legs (orthogonal basis).
    In case the "|" separator is present, tensor indices prior to "|" are
    considered contravariant, corresponding to OUTWARD tensor legs,  whereas
    tensor indices after "|" are considered covariant, corresponding to INWARD
    tensor legs (this distinction becomes essential in non-orthogonal bases).
(b) Valid symbolic tensor network formats:
    (a) <OutputTensor> = <InputTensor> * <InputTensor> * ... * <InputTensor>
    (b) <OutputTensor> += <InputTensor> * <InputTensor> * ... * <InputTensor>
    The number of tensors on the right-hand side is one or more.
**/

#ifndef EXATN_TENSOR_SYMBOL_HPP_
#define EXATN_TENSOR_SYMBOL_HPP_

#include "tensor_basic.hpp"
#include "tensor_leg.hpp"

#include <string>
#include <vector>

namespace exatn{

//Index label:
typedef struct{
 std::string label;      //alphanumeric_ index label
 LegDirection direction; //index variance (leg direction)
} IndexLabel;

inline bool is_letter(const char & ch){
 return ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'));
}

inline bool is_number(const char & ch){
 return (ch >= '0' && ch <= '9');
}

inline bool is_underscore(const char & ch){
 return (ch == '_');
}

inline bool is_space(const char & ch){
 return (ch == ' ');
}

inline bool is_conjugation_sign(const char & ch){
 return (ch == '+');
}

inline bool is_equal_sign(const char & ch){
 return (ch == '=');
}

inline bool is_plus_sign(const char & ch){
 return (ch == '+');
}

inline bool is_minus_sign(const char & ch){
 return (ch == '-');
}

inline bool is_multiply_sign(const char & ch){
 return (ch == '*');
}

/** Returns TRUE if the symbolic identifier is alphanumeric_ and starts with a letter. **/
inline bool is_alphanumeric(const std::string & identifier)
{
 if(identifier.empty()) return false;
 if(!is_letter(identifier[0])) return false;
 for(const char & ch: identifier){
  if(!(is_letter(ch) || is_number(ch) || is_underscore(ch))) return false;
 }
 return true;
}

/** Returns the string view range without leading and trailing spaces. **/
inline std::pair<int,int> trim_spaces_off(const std::string & str, //in: full string container
                                          std::pair<int,int> view) //in: input string view
{
 int sbeg = view.first;
 int send = view.second;
 if(sbeg <= send){
  while(send >= sbeg){ //remove trailing white spaces
   if(!is_space(str[send])) break;
   --send;
  }
  while(sbeg <= send){ //remove leading white spaces
   if(!is_space(str[sbeg])) break;
   ++sbeg;
  }
 }
 return std::make_pair(sbeg,send);
}

/** Generates a hexadecimal name for a tensor from an integer. **/
template <typename Integer>
std::string tensor_hex_name(Integer hash)
{
 static_assert(std::is_integral<Integer>::value,"#FATAL(tensor_hex_name): Non-integer type passed!");
 char digit[] = {'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};
 auto n = hash; if(n < 0) n = -n;
 std::string name("z");
 while(n > 0){
  auto m = n%16;
  name += digit[m];
  n /= 16;
 }
 return name;
}

/** Returns TRUE if the tensor parses as a valid symbolic tensor.
    The output function parameters will contain parsed tokens. **/
bool parse_tensor(const std::string & tensor,        //in: tensor as a string
                  std::string & tensor_name,         //out: tensor name
                  std::vector<IndexLabel> & indices, //out: tensor indices (labels)
                  bool & complex_conjugated);        //out: whether or not tensor appears complex conjugated

/** Returns TRUE if the tensor network parses as a valid symbolic tensor network.
    The output std::vector returns parsed symbolic tensors where element #0 is
    the output tensor of the tensor network. **/
bool parse_tensor_network(const std::string & network,         //in: tensor network as a string
                          std::vector<std::string> & tensors); //out: parsed (symbolic) tensors

/** Generates symbolic tensor contraction pattern from the digital tensor
    contraction pattern used by the contraction-based Tensor constructor:
     pattern[0..m-1] describes connectivity of dimensions of the left contracted tensor,
     pattern[m..m+n-1] decribes connectivity of dimensions of the right contracted tensor,
      where m and n are the ranks of the left and right contracted tensors, respectively.
     pattern[x] is a TensorLeg specifying the dimension of another tensor the described
      dimension is connected to, where the result tensor is tensor 0 while the left and
      right contracted tensors are tensors 1 and 2, respectively.
    **/
bool generate_contraction_pattern(const std::vector<numerics::TensorLeg> & pattern,
                                  unsigned int left_tensor_rank,
                                  unsigned int right_tensor_rank,
                                  std::string & symb_pattern);

} //namespace exatn

#endif //EXATN_TENSOR_SYMBOL_HPP_
