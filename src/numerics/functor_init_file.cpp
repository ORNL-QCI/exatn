/** ExaTN::Numerics: Tensor Functor: Initialization from a file
REVISION: 2020/12/04

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "functor_init_file.hpp"

#include "tensor_range.hpp"

#include "talshxx.hpp"

#include <iostream>
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

 auto read_func = [&](auto * tensor_body){
  std::ifstream tensor_file;
  tensor_file.open(filename_,std::fstream::in);
  if(tensor_file.is_open()){
   std::string format, tens_name, line;

   //Read the storage format:
   if(!std::getline(tensor_file,format)){
    std::cout << "#ERROR(exatn::numerics::FunctorInitFile): Invalid format of file " << filename_ << std::endl << std::flush;
    return 12;
   }

   //Read the stored tensor name:
   if(!std::getline(tensor_file,tens_name)){
    std::cout << "#ERROR(exatn::numerics::FunctorInitFile): Invalid format of file " << filename_ << std::endl << std::flush;
    return 11;
   }

   //Read the stored tensor shape:
   if(!std::getline(tensor_file,line)){
    std::cout << "#ERROR(exatn::numerics::FunctorInitFile): Invalid format of file " << filename_ << std::endl << std::flush;
    return 10;
   }
   std::stringstream shape(line);
   std::vector<DimExtent> tens_shape;
   DimExtent extent;
   while(shape >> extent) tens_shape.emplace_back(extent);
   unsigned int tens_rank = tens_shape.size();

   //Read the stored tensor signature:
   line.clear();
   if(!std::getline(tensor_file,line)){
    std::cout << "#ERROR(exatn::numerics::FunctorInitFile): Invalid format of file " << filename_ << std::endl << std::flush;
    return 9;
   }
   std::stringstream signa(line);
   std::vector<DimOffset> tens_signa;
   DimOffset base_offset;
   while(signa >> base_offset) tens_signa.emplace_back(base_offset);
   if(tens_signa.size() != tens_rank){
    std::cout << "#ERROR(exatn::numerics::FunctorInitFile): Inconsistent tensor shape/signature in file " << filename_ << std::endl << std::flush;
    return 8;
   }

   //Check rank/shape/signature:
   if(tens_rank != rank){
    std::cout << "#ERROR(exatn::numerics::FunctorInitFile): Tensor rank mismatch in file " << filename_ << std::endl << std::flush;
    return 7;
   }
   for(unsigned int i = 0; i < tens_rank; ++i){
    if(tens_shape[i] != extents[i]){
     std::cout << "#ERROR(exatn::numerics::FunctorInitFile): Tensor shape mismatch in file " << filename_ << std::endl << std::flush;
     return 6;
    }
   }
   for(unsigned int i = 0; i < tens_rank; ++i){
    if(tens_signa[i] != offsets[i]){
     std::cout << "#ERROR(exatn::numerics::FunctorInitFile): Tensor signature mismatch in file " << filename_ << std::endl << std::flush;
     return 5;
    }
   }

   //Read the stored tensor elements:
   std::size_t tens_volume = 0;
   line.clear();
   if(format == "dense"){
    while(std::getline(tensor_file,line)){
     std::stringstream inp(line);
     while(inp >> tensor_body[tens_volume]) tens_volume++;
     line.clear();
    }
   }else if(format == "list"){
    //`Finish
   }else{
    std::cout << "#ERROR(exatn::numerics::FunctorInitFile): Invalid storage format in file " << filename_ << std::endl << std::flush;
    return 3;
   }

   //Close file:
   tensor_file.close();
  }else{
   std::cout << "#ERROR(exatn::numerics::FunctorInitFile): File not found: " << filename_ << std::endl << std::flush;
   return 2;
  }
  return 0;
 };

 auto access_granted = false;
 {//Try REAL32:
  float * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return read_func(body);
 }

 {//Try REAL64:
  double * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return read_func(body);
 }

 {//Try COMPLEX32:
  std::complex<float> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return read_func(reinterpret_cast<float*>(body));
 }

 {//Try COMPLEX64:
  std::complex<double> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return read_func(reinterpret_cast<double*>(body));
 }

 std::cout << "#ERROR(exatn::numerics::FunctorInitFile): Unknown data kind in talsh::Tensor!" << std::endl << std::flush;
 return 1;
}

} //namespace numerics

} //namespace exatn
