/** ExaTN: Packable interface
REVISION: 2020/06/25

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef EXATN_PACKABLE_HPP_
#define EXATN_PACKABLE_HPP_

#include "byte_packet.h"

namespace exatn {

class Packable {

public:

 /** Packs the object into a plain byte packet. **/
 virtual void pack(BytePacket & byte_packet) const = 0;

 /** Unpacks the object from a plain byte packet. **/
 virtual void unpack(BytePacket & byte_packet) = 0;

};

} //namespace exatn

#endif //EXATN_PACKABLE_HPP_
