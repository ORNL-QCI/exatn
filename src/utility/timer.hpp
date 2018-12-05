/** ExaTN::Utility: Timers
REVISION: 2018/10/29

!Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
!Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef TIMER_HPP_
#define TIMER_HPP_

namespace exatn {

namespace utility {

double time_sys_sec();  // system time stamp in seconds (thread-global)
double time_high_sec(); // high-resolution time stamp in seconds

} // namespace utility

} // namespace exatn

#endif // TIMER_HPP_
