/** ExaTN::Numerics: Numerical server
REVISION: 2019/05/27

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef EXATN_NUMERICS_NUM_SERVER_HPP_
#define EXATN_NUMERICS_NUM_SERVER_HPP_

#include "tensor_basic.hpp"
#include "space_register.hpp"
#include "tensor.hpp"
#include "tensor_operation.hpp"
#include "tensor_network.hpp"

#include "tensor_method.hpp"
#include "Identifiable.hpp"

#include <memory>
#include <string>
#include <map>

using exatn::Identifiable;

namespace exatn{

namespace numerics{

class NumServer{

public:

 NumServer() = default;
 NumServer(const NumServer &) = delete;
 NumServer & operator=(const NumServer &) = delete;
 NumServer(NumServer &&) noexcept = default;
 NumServer & operator=(NumServer &&) noexcept = default;
 ~NumServer() = default;

 /** Registers an external tensor method. **/
 void registerTensorMethod(std::shared_ptr<TensorMethod<Identifiable>> method);

 /** Retrieves a registered external tensor method. **/
 std::shared_ptr<TensorMethod<Identifiable>> getTensorMethod(const std::string & tag);

 /** Registers an external data packet. **/
 void registerExternalData(const std::string & tag, std::shared_ptr<BytePacket> packet);

 /** Retrieves a registered external data packet. **/
 std::shared_ptr<BytePacket> getExternalData(const std::string & tag);

private:

 SpaceRegister space_register_; //register of vector spaces and their named subspaces
 std::unordered_map<std::string,SpaceId> subname2id_; //maps a subspace name to its parental vector space id

 std::map<std::string,std::shared_ptr<TensorMethod<Identifiable>>> ext_methods_; //external tensor methods
 std::map<std::string,std::shared_ptr<BytePacket>> ext_data_; //external data

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_NUM_SERVER_HPP_
