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

class empty{
};

template<typename>
struct dummy{
};

template<template<typename> class TT>
struct is_dummy :
	std::is_same<dummy<void>, TT<void>>{
};

struct inner_t{};

template<template<typename> class TT>
inline constexpr bool is_dummy_v = is_dummy<TT>::value;

template<typename T>
class materialized_ptr{
	T value;
public:
	typedef T value_type;
	typedef T* pointer;
	typedef T& reference;

	pointer operator->() const& noexcept{
		return const_cast<pointer>(&value);
	}
	// pointer operator->() && = delete;
	reference operator*() const& noexcept{
		return const_cast<reference>(value);
	}
	value_type operator*() && noexcept{
		return std::move(const_cast<reference>(value));
	}

	materialized_ptr() requires std::is_default_constructible_v<T> : value(){
	}
	template<typename U>
	materialized_ptr(U &&value) :
		value(std::forward<U>(value)){
	}
	materialized_ptr(materialized_ptr&&) = default;
	materialized_ptr(const materialized_ptr&) = delete;
	materialized_ptr& operator=(materialized_ptr&&) = default;
	materialized_ptr& operator=(const materialized_ptr&) = delete;
};

template<typename T>
class materialized_ptr<T&>{
	std::reference_wrapper<T> ref;
public:
	typedef T value_type;
	typedef T* pointer;
	typedef T& reference;

	pointer operator->() const noexcept{
		return &ref.get();
	}
	reference operator*() const noexcept{
		return ref.get();
	}

	// TODO: use `requires` instead in C++20
	materialized_ptr(T &ref) : ref(ref){}
	materialized_ptr(materialized_ptr&&) = default;
	materialized_ptr(const materialized_ptr&) = delete;
	materialized_ptr& operator=(materialized_ptr&&) = default;
	materialized_ptr& operator=(const materialized_ptr&) = delete;
};

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
