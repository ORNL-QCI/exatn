/** ExaTN: Tensor Runtime: Tensor network executor: Linear memory allocator
REVISION: 2022/01/03

Copyright (C) 2018-2022 Dmitry Lyakh
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

 Linear memory moving window (----->):

 (a) front >= back:
 ____________________________________
 |          |                |       |
 |  FREE    |    OCCUPIED    | FREE  |
 |          |                |       |
 |__________|________________|_______|
 begin     back            front   end

 (b) front < back:
 ____________________________________
 |          |              |         |
 | OCCUPIED |      FREE    | OCCUPIED|
 |          |              |         |
 |__________|______________|_________|
 begin    front           back     end

**/

#ifndef EXATN_RUNTIME_LINEAR_MEMORY_HPP_
#define EXATN_RUNTIME_LINEAR_MEMORY_HPP_

#include "errors.hpp"

class LinearMemoryPool {

public:

 LinearMemoryPool(void * base_ptr,
                  std::size_t total_size,
                  std::size_t alignment):
  base_ptr_(base_ptr), total_size_(total_size), alignment_(alignment),
  front_(base_ptr), back_(base_ptr)
 {
  assert(reinterpret_cast<std::size_t>(base_ptr_) % alignment_ == 0);
 }

 std::size_t occupiedSize() const {
  const std::size_t fptr = reinterpret_cast<std::size_t>(front_);
  const std::size_t bptr = reinterpret_cast<std::size_t>(back_);
  if(fptr >= bptr) return (fptr - bptr);
  return (total_size_ - bptr + fptr);
 }

 void * acquireMemory(std::size_t mem_size) {
  assert(mem_size > 0);
  mem_size = (mem_size - (mem_size % alignment_)) + alignment_;
  if(occupiedSize() + mem_size > total_size_) return nullptr;
  void * mem_ptr = front_;
  std::size_t left_forward = (total_size_ - reinterpret_cast<std::size_t>(front_));
  if(left_forward > mem_size){
   front_ = (void*)((char*)front_ + mem_size);
  }else{
   front_ = (void*)((char*)base_ptr_ + (mem_size - left_forward));
  }
  return mem_ptr;
 }

 void releaseMemory(void * back_ptr) {
  assert(reinterpret_cast<std::size_t>(back_ptr) % alignment_ == 0);
  const auto preceding_size = occupiedSize();
  back_ = back_ptr;
  assert(occupiedSize() < preceding_size);
  return;
 }

 void restorePreviousFront(void * front) {
  front_ = front;
  return;
 }

 void * getFront() const {
  return front_;
 }

 void * getBack() const {
  return back_;
 }

protected:

 void * base_ptr_;
 std::size_t total_size_;
 std::size_t alignment_;
 void * front_;
 void * back_;
};

#endif //EXATN_RUNTIME_LINEAR_MEMORY_HPP_
