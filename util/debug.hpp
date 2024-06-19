#ifndef _ANN_UTIL_DEBUG_HPP
#define _ANN_UTIL_DEBUG_HPP

#include <cstdio>

namespace ANN::util{

template<typename ...Args>
inline void debug_output(Args ...args)
{
#ifdef DEBUG_OUTPUT
	fprintf(stderr, args...);
#else
	((void)(args), ...);
#endif // DEBUG_OUTPUT
}

} // namespace ANN::util

// extern parlay::sequence<size_t> per_visited;
// extern parlay::sequence<size_t> per_eval;
// extern parlay::sequence<size_t> per_size_C;

#endif // _ANN_UTIL_DEBUG_HPP
