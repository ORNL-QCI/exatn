/** ExaTN::Numerics: Numerical server
REVISION: 2019/05/07

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef EXATN_NUMERICS_NUM_SERVER_HPP_
#define EXATN_NUMERICS_NUM_SERVER_HPP_

#include "tensor_basic.hpp"
#include "space_register.hpp"
#include "tensor_factory.hpp"
#include "tensor_network.hpp"

#include "tensor_method.hpp"
#include "Identifiable.hpp"

#include <string>
#include <unordered_map>

using exatn::Identifiable;

namespace exatn{

namespace numerics{

class NumServer{ //singleton

public:

 NumServer() = default;
 NumServer(const NumServer &) = delete;
 NumServer & operator=(const NumServer &) = delete;
 NumServer(NumServer &&) noexcept = default;
 NumServer & operator=(NumServer &&) noexcept = default;
 ~NumServer() = default;

 virtual void addTensorMethod(std::shared_ptr<TensorMethod<Identifiable>> method) {
   methods.insert({method->name(), method});
 }

 virtual BytePacket getExternalData(const std::string& tag) {
   return extData[tag];
 }

private:

 SpaceRegister space_register_; //register of vector spaces and their named subspaces
 std::unordered_map<std::string,SpaceId> subname2id_; //maps a subspace name to its parental vector space id

protected:

  std::map<std::string, std::shared_ptr<TensorMethod<Identifiable>>> methods;

  std::map<std::string, BytePacket> extData;
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_NUM_SERVER_HPP_
