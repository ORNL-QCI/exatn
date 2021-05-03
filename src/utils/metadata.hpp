/** ExaTN: Short metadata
REVISION: 2021/04/27

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Class Metadata serves for quick storage and retrieval of small key-value metadata.
     Key is always alphanumeric_, value can be of any simple type:
     integer/long (signed or unsigned), float/double, bool, and string.
**/

#ifndef EXATN_METADATA_HPP_
#define EXATN_METADATA_HPP_

#include <string>

#include "errors.hpp"

namespace exatn{

class Metadata{

public:

 /** Appends a key:value pair to the metadata.
     The key must be alphanumeric_. **/
 template<typename ValueType>
 inline void appendKeyValue(const std::string & key,
                            const ValueType & value);

 /** Retrieves a value from the metadata by its key. **/
 template<typename ValueType>
 inline bool retrieveValue(const std::string & key,
                           ValueType & value) const;

 /** Clears metadata. **/
 inline void clear(){
  data_.clear();
 }

private:

 std::string data_; //metadata storage
};


template<typename T>
inline T convertStringToValue(const std::string & str)
{
 return T(str);
}

template<>
inline int convertStringToValue<int>(const std::string & str)
{
 return std::stoi(str);
}

template<>
inline long convertStringToValue<long>(const std::string & str)
{
 return std::stol(str);
}

template<>
inline long long convertStringToValue<long long>(const std::string & str)
{
 return std::stoll(str);
}

template<>
inline unsigned long convertStringToValue<unsigned long>(const std::string & str)
{
 return std::stoul(str);
}

template<>
inline unsigned int convertStringToValue<unsigned int>(const std::string & str)
{
 return static_cast<unsigned int>(std::stoul(str));
}

template<>
inline unsigned long long convertStringToValue<unsigned long long>(const std::string & str)
{
 return std::stoull(str);
}

template<>
inline float convertStringToValue<float>(const std::string & str)
{
 return std::stof(str);
}

template<>
inline double convertStringToValue<double>(const std::string & str)
{
 return std::stod(str);
}

template<>
inline bool convertStringToValue<bool>(const std::string & str)
{
 assert(str.length() == 1);
 assert(str[0] == 'T' || str[0] == 'F');
 return (str[0] == 'T');
}

template<>
inline std::string convertStringToValue<std::string>(const std::string & str)
{
 return str;
}


template<typename ValueType>
inline void Metadata::appendKeyValue(const std::string & key, const ValueType & value)
{
 data_ += "{" + key + ":" + std::to_string(value) + "}";
 return;
}

template<typename ValueType>
inline bool Metadata::retrieveValue(const std::string & key, ValueType & value) const
{
 auto pos = data_.find(key+":");
 if(pos != std::string::npos){
  const auto key_len = key.length();
  const auto value_beg_pos = pos + key_len + 1;
  const auto value_end_pos = data_.find("}",value_beg_pos);
  const auto value_len = value_end_pos - value_beg_pos;
  value = convertStringToValue<ValueType>(data_.substr(value_beg_pos,value_len));
  return true;
 }
 return false;
}

} //namespace exatn

#endif //EXATN_METADATA_HPP_
