#ifndef _ANN_UTIL_INTRIN_HPP
#define _ANN_UTIL_INTRIN_HPP

#include <cstdint>
#include <iterator>
#include <type_traits>
#include "util/util.hpp"
#include "custom/custom.hpp"

namespace ANN::util{

namespace detail{

template<typename T, typename=void>
struct has_for_each : std::false_type{
};
template<typename T>
struct has_for_each<T, std::void_t<
	decltype(std::declval<T>().for_each(identity{}))
>> : std::true_type{
};
template<typename T>
inline constexpr bool has_for_each_v = has_for_each<T>::value;

template<typename T, typename=void>
struct has_iter_each : std::false_type{
};
template<typename T>
struct has_iter_each<T, std::void_t<
	decltype(std::declval<T>().for_each(identity{}))
>> : std::true_type{
};
template<typename T>
inline constexpr bool has_iter_each_v = has_iter_each<T>::value;

template<typename T>
auto declbegin(T &&r)
{
	using std::begin;
	return begin(r);
}
template<typename T>
using begin_t = decltype(declbegin(std::declval<T>()));

template<typename T, typename=void>
struct is_randomly_iterable : std::false_type{
};
// TODO: use `random_access_iterator` in C++20
template<typename T>
struct is_randomly_iterable<T, std::enable_if_t<
	std::is_base_of_v<
		std::random_access_iterator_tag,
		typename std::iterator_traits<begin_t<T>>::iterator_category
	>
>> : std::true_type{
};
template<typename T>
inline constexpr bool is_randomly_iterable_v = is_randomly_iterable<T>::value;

} // namespace detail

template<class R, class F>
void iter_each(R &&r, F &&f)
{
	if constexpr(detail::has_iter_each_v<R>)
		r.iter_each(std::forward<F>(f));
	else for(auto &&v : r) // TODO: use std::views::as_rvalue in C++23
		f(std::forward<decltype(v)>(v));
}

template<class R, class F>
void for_each(R &&r, F &&f)
{
	using cm = custom<typename lookup_custom_tag<R>::type>;
	// TODO: use `requires` in C++20
	if constexpr(detail::has_for_each_v<R>)
		r.for_each(std::forward<F>(f));
	else if constexpr(detail::is_randomly_iterable_v<R>)
	{
		// TODO: use `std::ranges::*` in C++20
		using std::begin, std::size;
		auto it = begin(r);
		auto n = size(r);
		cm::parallel_for(0, n, [&](auto i){f(it[i]);});
	}
	else iter_each(std::forward<R>(r), std::forward<F>(f));
}

template<class C, class R, class F=identity>
C to(R &&r)
{
	static_assert(std::is_convertible_v<R,C>);
	return C(std::forward<R>(r));
}

/*
template<class C, class R, class F=identity>
C to(R &&r, F &&f={})
{
	if constexpr(requires(size() and reserve()))
	{
		reserve(size())
	}
	handle move semantics
	- direct construct
	- from_range_t
	- iter-pair
	- push_back/insert
}
template<class C, class F=identity>
C tabulate(size_t n, F &&f={})
{
	return to<C>(delayed_seq(n,f));
}
*/
} // namespace ANN::util

#endif // _ANN_UTIL_INTRIN_HPP
