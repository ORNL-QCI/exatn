/** ExaTN::Numerics: Tensor Functor: Initialization from a file
REVISION: 2020/12/03

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "functor_init_file.hpp"

#include "tensor_range.hpp"

#include "talshxx.hpp"

#include <fstream>

namespace exatn{

namespace numerics{

FunctorInitFile::FunctorInitFile(const std::string & filename):
 filename_(filename)
{
}


void FunctorInitFile::pack(BytePacket & packet)
{
 unsigned int filename_len = filename_.length();
 appendToBytePacket(&packet,filename_len);
 while(filename_len > 0) appendToBytePacket(&packet,filename_[--filename_len]);
 return;
}


void FunctorInitFile::unpack(BytePacket & packet)
{
 unsigned int filename_len;
 extractFromBytePacket(&packet,filename_len);
 if(filename_len > 0){
  filename_.resize(filename_len);
  for(unsigned int i = 0; i < filename_len; ++i) extractFromBytePacket(&packet,filename_[i]);
 }else{
  filename_.clear();
 }
 return;
}


int FunctorInitFile::apply(talsh::Tensor & local_tensor) //tensor slice (in general)
{
 unsigned int rank;
 const auto * extents = local_tensor.getDimExtents(rank); //rank is returned by reference
 const auto tensor_volume = local_tensor.getVolume(); //volume of the given tensor slice
 const auto & offsets = local_tensor.getDimOffsets(); //base offsets of the given tensor slice

 auto read_func = [&](const auto * tensor_body){
  std::fstream tensor_file;
  tensor_file.open(filename_,std::fstream::in);
  if(tensor_file.is_open()){
   //`Finish
  }else{
   std::cout << "#ERROR(exatn::numerics::FunctorInitFile): File not found: " << filename_ << std::endl << std::flush;
   assert(false);
  }
  return;
 };

 auto access_granted = false;
 {//Try REAL32:
  float * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   read_func(body);
   return 0;
  }
 }

 {//Try REAL64:
  double * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   read_func(body);
   return 0;
  }
 }

 {//Try COMPLEX32:
  std::complex<float> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   read_func(body);
   return 0;
  }
 }

 {//Try COMPLEX64:
  std::complex<double> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   read_func(body);
   return 0;
  }
 }

 std::cout << "#ERROR(exatn::numerics::FunctorInitFile): Unknown data kind in talsh::Tensor!" << std::endl << std::flush;
 return 1;
}

} //namespace numerics

} //namespace exatn
