/** ExaTN::Numerics: Tensor leg (connection)
REVISION: 2019/10/21

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) A tensor leg associates a tensor mode with a mode in another tensor
     by carrying the id of another tensor, its specific mode (position),
     and direction of the association.
**/

#ifndef EXATN_NUMERICS_TENSOR_LEG_HPP_
#define EXATN_NUMERICS_TENSOR_LEG_HPP_

#include "tensor_basic.hpp"

#include <iostream>
#include <vector>

namespace exatn{

namespace numerics{

inline LegDirection reverseLegDirection(LegDirection dir){
 if(dir == LegDirection::INWARD) return LegDirection::OUTWARD;
 if(dir == LegDirection::OUTWARD) return LegDirection::INWARD;
 return LegDirection::UNDIRECT;
}


class TensorLeg{
public:

 /** Create a tensor leg by specifying the id of the connected tensor [0:*] and its corresponding dimension [0:*]. **/
 TensorLeg(unsigned int tensor_id,
           unsigned int dimensn_id,
           LegDirection direction = LegDirection::UNDIRECT);

 TensorLeg(const TensorLeg & tens_leg) = default;
 TensorLeg & operator=(const TensorLeg & tens_leg) = default;
 TensorLeg(TensorLeg && tens_leg) noexcept = default;
 TensorLeg & operator=(TensorLeg && tens_leg) noexcept = default;
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

 /** Reverse the direction of the tensor leg. **/
 void reverseDirection();

private:

 unsigned int tensor_id_;  //id of the connected tensor [0..*]
 unsigned int dimensn_id_; //dimension id in the connected tensor [0..*]
 LegDirection direction_;  //direction of the leg: {UNDIRECT, INWARD, OUTWARD}, defaults to UNDIRECT
};


/** Returns true if two vectors of tensor legs are congruent, that is,
    they have the same size and direction of each tensor leg. **/
bool tensorLegsAreCongruent(const std::vector<TensorLeg> * legs0,
                            const std::vector<TensorLeg> * legs1);

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_LEG_HPP_
