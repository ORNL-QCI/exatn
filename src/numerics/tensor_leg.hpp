/** ExaTN::Numerics: Tensor leg (connection)
REVISION: 2018/12/18

Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef TENSOR_LEG_HPP_
#define TENSOR_LEG_HPP_

#include "tensor_basic.hpp"

#include <assert.h>

#include <iostream>

namespace exatn{

namespace numerics{

class TensorLeg{
public:

 /** Create a tensor leg by specifying the id of the connected tensor [0:*] and its corresponding dimension [0:*]. **/
 explicit TensorLeg(unsigned int tensor_id,
                    unsigned int dimensn_id,
                    LegDirection direction = LegDirection::UNDIRECT);

 TensorLeg(const TensorLeg & tens_leg) = default;
 TensorLeg & operator=(const TensorLeg & tens_leg) = default;
 TensorLeg(TensorLeg && tens_leg) = default;
 TensorLeg & operator=(TensorLeg && tens_leg) = default;
 virtual ~TensorLeg() = default;

 /** Print. **/
 void printIt() const;

 /** Return the connected tensor id: [0..*]. **/
 unsigned int getTensorId() const;

 /** Return the connected tensor dimension id: [0..*]. **/
 unsigned int getDimensionId() const;

 /** Return the direction of the tensor leg. **/
 LegDirection getDirection() const;

 /** Reset the tensor leg to another connection. **/
 void resetConnection(unsigned int tensor_id,
                      unsigned int dimensn_id,
                      LegDirection direction = LegDirection::UNDIRECT);

 /** Reset only the tensor id in the tensor leg. **/
 void resetTensorId(unsigned int tensor_id);

 /** Reset only the tensor dimension id in the tensor leg. **/
 void resetDimensionId(unsigned int dimensn_id);

 /** Reset the direction of the tensor leg. **/
 void resetDirection(LegDirection direction);

private:

 unsigned int tensor_id_;  //id of the connected tensor [0..*]
 unsigned int dimensn_id_; //dimension id in the connected tensor [0..*]
 LegDirection direction_;  //direction of the leg: {UNDIRECT, INWARD, OUTWARD}
};

} //namespace numerics

} //namespace exatn

#endif //TENSOR_LEG_HPP_
