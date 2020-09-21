/** ExaTN: Error handling
REVISION: 2020/09/21

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef EXATN_ERRORS_HPP_
#define EXATN_ERRORS_HPP_

#include "timers.hpp"

#include <string>

namespace exatn {

inline void fatal_error()
{
 std::cout << "An error occurred in ExaTN!" << std::endl << std::flush;
 std::cerr << "An error occurred in ExaTN!" << std::endl << std::flush;
 std::abort();
}

inline void fatal_error(const std::string & error_msg)
{
 std::cout << "#ERROR: " << error_msg << std::endl << std::flush;
 std::cerr << "#ERROR: " << error_msg << std::endl << std::flush;
 return fatal_error();
}

inline void make_sure(bool condition)
{
 if(!condition) fatal_error();
 return;
}

inline void make_sure(bool condition,
                      const std::string & error_msg)
{
 if(!condition) fatal_error(error_msg);
 return;
}

} //namespace exatn

#endif //EXATN_ERRORS_HPP_
