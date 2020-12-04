/** ExaTN::Numerics: Tensor Functor: Prints a tensor to a file or standard output
REVISION: 2020/12/04

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "functor_print.hpp"

#include "tensor_range.hpp"

#include "talshxx.hpp"

#include <iostream>
#include <fstream>

namespace exatn{

namespace numerics{

FunctorPrint::FunctorPrint(const std::string & filename):
 filename_(filename)
{
}


void FunctorPrint::pack(BytePacket & packet)
{
 unsigned int filename_len = filename_.length();
 appendToBytePacket(&packet,filename_len);
 while(filename_len > 0) appendToBytePacket(&packet,filename_[--filename_len]);
 return;
}


void FunctorPrint::unpack(BytePacket & packet)
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


int FunctorPrint::apply(talsh::Tensor & local_tensor) //tensor slice (in general)
{
 unsigned int rank = 0;
 const auto * extents = local_tensor.getDimExtents(rank); //rank is returned by reference
 std::size_t tensor_volume = local_tensor.getVolume(); //volume of the given tensor slice
 const auto & offsets = local_tensor.getDimOffsets(); //base offsets of the given tensor slice
 const auto element_type = local_tensor.getElementType(); //tensor element type

 auto print_func = [&](auto * tensor_body){
  std::ofstream tensor_file;
  std::ostream * output = nullptr;
  if(filename_.length() > 0){
   tensor_file.open(filename_,std::fstream::out);
   if(tensor_file.is_open()) output = &tensor_file;
  }else{
   output = &std::cout;
  }

  if(output != nullptr){
   *output << "dense" << std::endl;
   *output << "tensor" << std::endl;
   for(unsigned int i = 0; i < rank; ++i) *output << " " << extents[i];
   *output << std::endl;
   for(unsigned int i = 0; i < rank; ++i) *output << " " << offsets[i];
   *output << std::endl;
   if(element_type == talsh::COMPLEX32 || element_type == talsh::COMPLEX64) tensor_volume *= 2;
   *output << std::scientific;
   for(std::size_t i = 0; i < tensor_volume; ++i){
    *output << " " << tensor_body[i];
    if(i%16 == 15) *output << std::endl;
   }
   if(tensor_volume%16 != 0) *output << std::endl;
   if(filename_.length() > 0) tensor_file.close();
  }else{
   std::cout << "#ERROR(exatn::numerics::FunctorPrint): Output failed!" << std::endl << std::flush;
   return 2;
  }
  return 0;
 };

 auto access_granted = false;
 {//Try REAL32:
  float * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return print_func(body);
 }

 {//Try REAL64:
  double * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return print_func(body);
 }

 {//Try COMPLEX32:
  std::complex<float> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return print_func(reinterpret_cast<float*>(body));
 }

 {//Try COMPLEX64:
  std::complex<double> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return print_func(reinterpret_cast<double*>(body));
 }

 std::cout << "#ERROR(exatn::numerics::FunctorPrint): Unknown data kind in talsh::Tensor!" << std::endl << std::flush;
 return 1;
}

} //namespace numerics

} //namespace exatn
