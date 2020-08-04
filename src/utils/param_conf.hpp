/** ExaTN: Parameter Configuration
REVISION: 2020/04/27

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef EXATN_PARAM_CONF_HPP_
#define EXATN_PARAM_CONF_HPP_

#include <map>
#include <string>

#include <cstdint>

namespace exatn {

class ParamConf {

public:

 ParamConf() = default;
 ParamConf(const ParamConf &) = default;
 ParamConf & operator=(const ParamConf &) = default;
 ParamConf(ParamConf &&) noexcept = default;
 ParamConf & operator=(ParamConf &&) noexcept = default;
 ~ParamConf() = default;

 bool setParameter(const std::string & name,
                   const int64_t value)
 {
  auto res = ints_.emplace(std::make_pair(name,value));
  return res.second;
 }

 bool setParameter(const std::string & name,
                   const float value)
 {
  auto res = reals_.emplace(std::make_pair(name,static_cast<double>(value)));
  return res.second;
 }

 bool setParameter(const std::string & name,
                   const double value)
 {
  auto res = reals_.emplace(std::make_pair(name,value));
  return res.second;
 }

 bool setParameter(const std::string & name,
                   const std::string & value)
 {
  auto res = strings_.emplace(std::make_pair(name,value));
  return res.second;
 }

 bool getParameter(const std::string & name,
                   int64_t * value) const
 {
  auto iter = ints_.find(name);
  if(iter == ints_.end()) return false;
  assert(value != nullptr);
  *value = iter->second;
  return true;
 }

 bool getParameter(const std::string & name,
                   float * value) const
 {
  auto iter = reals_.find(name);
  if(iter == reals_.end()) return false;
  assert(value != nullptr);
  *value = static_cast<float>(iter->second);
  return true;
 }

 bool getParameter(const std::string & name,
                   double * value) const
 {
  auto iter = reals_.find(name);
  if(iter == reals_.end()) return false;
  assert(value != nullptr);
  *value = iter->second;
  return true;
 }

 bool getParameter(const std::string & name,
                   std::string & value) const
 {
  auto iter = strings_.find(name);
  if(iter == strings_.end()) return false;
  value = iter->second;
  return true;
 }

private:

 std::map<std::string,int64_t> ints_;
 std::map<std::string,double> reals_;
 std::map<std::string,std::string> strings_;
};

} //namespace exatn

#endif //EXATN_PARAM_CONF_HPP_
