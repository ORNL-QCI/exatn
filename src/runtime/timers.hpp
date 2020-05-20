/** ExaTN: Timers
REVISION: 2020/05/20

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef EXATN_TIMERS_HPP_
#define EXATN_TIMERS_HPP_

#include <chrono>

namespace exatn{

class Timer{
public:

 Timer(): start_(0.0), finish_(0.0), started_(false) {}

 Timer(const Timer &) = default;
 Timer & operator=(const Timer &) = default;
 Timer(Timer &&) noexcept = default;
 Timer & operator=(Timer &&) noexcept = default;
 ~Timer() = default;

 inline bool start(){
  if(!started_){
   start_ = timeInSecHR();
   started_ = true;
   return true;
  }
  return false;
 }

 inline bool stop(){
  if(started_){
   finish_ = timeInSecHR();
   started_ = false;
   return true;
  }
  return false;
 }

 inline double getDuration() const{
  return (finish_ - start_);
 }

 inline double getStartTime() const{
  return start_;
 }

 inline double getFinishTime() const{
  return finish_;
 }

 static inline double timeInSec(){
  auto stamp = std::chrono::system_clock::now(); //current time point
  auto durat = std::chrono::duration<double>(stamp.time_since_epoch()); //duration (sec) since the begining of the clock
  return durat.count(); //number of seconds
 }

 static inline double timeInSecHR()
 {
  auto stamp = std::chrono::high_resolution_clock::now(); //current time point
  auto durat = std::chrono::duration<double>(stamp.time_since_epoch()); //duration (sec) since the begining of the clock
  return durat.count(); //number of seconds
 }

private:

 double start_;
 double finish_;
 bool started_;
};

} //namespace exatn

#endif //EXATN_TIMERS_HPP_
