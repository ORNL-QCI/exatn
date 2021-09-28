/** ExaTN: Error handling
REVISION: 2021/09/27

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef EXATN_ERRORS_HPP_
#define EXATN_ERRORS_HPP_

#include <iostream>
#include <string>

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

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
 if(!condition) return fatal_error();
 return;
}

inline void make_sure(bool condition,
                      const std::string & error_msg)
{
 if(!condition) return fatal_error(error_msg);
 return;
}

inline void print_variadic_pack()
{
 return;
}

template <typename Arg, typename... Args>
inline void print_variadic_pack(Arg&& arg, Args&&... args)
{
 std::cout << std::forward<Arg>(arg) << " ";
 return print_variadic_pack(std::forward<Args>(args)...);
}

} //namespace exatn

#endif //EXATN_ERRORS_HPP_
