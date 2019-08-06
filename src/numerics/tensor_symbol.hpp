/** ExaTN: Numerics: Symbolic tensor processing
REVISION: 2019/08/06

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef EXATN_TENSOR_SYMBOL_HPP_
#define EXATN_TENSOR_SYMBOL_HPP_

#include "tensor_basic.hpp"

#include <string>
#include <vector>

namespace exatn{

typedef struct{
 std::string label;
 LegDirection direction;
} IndexLabel;

/** Returns TRUE if the tensor is a valid symbolic tensor.
    The output function parameters will contain the associated info. **/
bool parse_tensor(const std::string & tensor,        //in: tensor as a string
                  std::string & tensor_name,         //out: tensor name
                  std::vector<IndexLabel> & indices, //out: tensor indices (labels)
                  bool & complex_conjugated);        //out: whether or not tensor appears complex conjugated

} //namespace exatn

#endif //EXATN_TENSOR_SYMBOL_HPP_
