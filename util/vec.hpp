#ifndef _ANN_UTIL_VEC_HPP
#define _ANN_UTIL_VEC_HPP

#include <cassert>
#include <cstdint>
#include <iterator>
#include <utility>
#include <type_traits>
#include "custom/custom.hpp"
#include "util.hpp"

namespace ANN::util{

template<typename C>
class vec
{
	using coord_t = C;
	coord_t coord;
public:
	using elem_t = typename coord_t::value_type;

	vec() = default;
	vec(const vec&) = default;
	vec(vec &&) noexcept = default;

	template<typename A, typename ...Args, typename=std::enable_if_t<
		!std::is_same_v<std::decay_t<A>, inner_t> &&
		!std::is_same_v<std::decay_t<A>, vec>
	>>
	vec(A &&a, Args &&...args) :
		coord(std::forward<A>(a), std::forward<Args>(args)...){
	}

	template<typename U>
	vec(inner_t, const U &src, uint32_t dim) : coord(dim){
		for(uint32_t i=0; i<dim; ++i)
			coord[i] = src[i];
	}

	vec& operator=(const vec&) = default;
	vec& operator=(vec &&) noexcept = default;

	const elem_t* data() const{
		return std::data(coord);
	}
	auto size() const{
		return coord.size();
	}
	decltype(auto) operator[](size_t idx) const{
		return coord[idx];
	}

	vec& operator+=(const vec &v){
		assert(size()==v.size());
		for(size_t i=0; i<size(); ++i)
			coord[i] += v[i];
		return *this;
	}

	vec& operator-=(const vec &v){
		assert(size()==v.size());
		for(size_t i=0; i<size(); ++i)
			coord[i] -= v[i];
		return *this;
	}

	template<typename T>
	vec& operator*=(const T &val){
		for(auto &e : coord) e *= val;
		return *this;
	}

	template<typename T>
	vec& operator/=(const T &val){
		for(auto &e : coord) e /= val;
		return *this;
	}

	friend vec operator-(const vec &v){
		auto res = v;
		for(auto &e : res.coord) e = -e;
		return res;
	}
};

template<typename C>
inline vec<C> operator+(const vec<C> &lhs, const vec<C> &rhs){
	auto res = lhs;
	res += rhs;
	return res;
}

template<typename C>
inline vec<C> operator-(const vec<C> &lhs, const vec<C> &rhs){
	auto res = lhs;
	res -= rhs;
	return res;
}

template<typename C, typename T>
inline vec<C> operator*(const vec<C> &v, const T &val){
	auto res = v;
	res *= val;
	return res;
}
template<typename C, typename T>
inline vec<C> operator*(const T &val, const vec<C> &v){
	return v*val;
}

template<typename C, typename T>
inline vec<C> operator/(const vec<C> &v, const T &val){
	auto res = v;
	res /= val;
	return res;
}

} // namespace ANN::util

#endif // _ANN_UTIL_VEC_HPP
