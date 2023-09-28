#ifndef _ANN_UTIL_HPP
#define _ANN_UTIL_HPP

#include <utility>
#include <type_traits>

namespace ANN{
namespace util{

template<typename Nid>
struct conn{
	float d;	// distance stays as the first member
	Nid u;

	constexpr bool operator<(const conn &rhs) const{
		return d<rhs.d;
	}

	constexpr bool operator>(const conn &rhs) const{
		return d>rhs.d;
	}
};

/*
struct near{
	template<typename Nid>
	constexpr bool operator()(const conn<Nid> &lhs, const conn<Nid> &rhs) const{
		return lhs.d<rhs.d;
	}
};

struct far{
	template<typename Nid>
	constexpr bool operator()(const conn<Nid> &lhs, const conn<Nid> &rhs) const{
		return lhs.d>rhs.d;
	}
};

struct same_node{
	template<typename Nid>
	constexpr bool operator()(const conn<Nid> &lhs, const conn<Nid> &rhs) const{
		return lhs.u==rhs.u;
	}
};

struct hash_node{
	template<typename Nid>
	constexpr auto operator()(const conn<Nid> &p) const{
		return std::hash<Nid>{}(p.u);
	}
};
*/

// TODO: use `std::identity` in C++20
class identity
{
	template<typename T>
	constexpr T&& operator()(T &&x) const noexcept{
		return std::forward<T>(x);
	}
};

template<typename>
struct dummy{
};

template<template<typename> class TT>
struct is_dummy :
	std::is_same<dummy<void>, TT<void>>{
};

template<template<typename> class TT>
inline constexpr bool is_dummy_v = is_dummy<TT>::value;

template<class R, typename=void, typename ...Ts>
struct is_direct_list_initializable_impl : std::false_type{
};

template<class R, typename ...Ts>
struct is_direct_list_initializable_impl<R,std::void_t<
	decltype(R{std::declval<Ts>()...})
>,Ts...> : std::true_type{
};

template<class R, typename ...Ts>
using is_direct_list_initializable = 
	is_direct_list_initializable_impl<R, void, Ts...>;

template<typename R, typename... Ts>
constexpr bool is_direct_list_initializable_v = 
	is_direct_list_initializable<R, Ts...>::value;

} // namespace util
} // namespace ANN

#endif // _ANN_UTIL_HPP