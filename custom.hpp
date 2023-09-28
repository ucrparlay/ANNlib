#ifndef _ANN_CUSTOM_HPP
#define _ANN_CUSTOM_HPP

#include <cstdint>
#include <vector>
#include <algorithm>
#include <memory>
#include <execution>
#include <functional>
#include <type_traits>
#include "util/seq.hpp"

namespace ANN{
namespace external{

class custom_tag_undef;

auto def_custom_tag();

template<typename T, typename=void>
struct lookup_custom_tag_impl{
	using type = custom_tag_undef;
};

template<typename T>
struct lookup_custom_tag_impl<T,std::void_t<decltype(def_custom_tag(),bool{})>>{
	using type = decltype(def_custom_tag());
};

template<typename T=void>
using lookup_custom_tag = lookup_custom_tag_impl<T>;

template<class Tag>
class custom;

template<>
class custom<custom_tag_undef>
{
public:
	template<typename T>
	using alloc = std::allocator<T>;

	template<typename T>
	using seq = std::vector<T>;

	template <typename F>
	static void parallel_for(
		size_t start, size_t end, F f,
		long granularity=0, bool conservative=false)
	{
		(void)granularity, (void)conservative;

		util::delayed_seq a(end-start, [=](size_t i){return i+start;});
		std::for_each(std::execution::par, a.begin(), a.end(), f);
	}

	static uint64_t hash64(uint64_t x)
	{
		return std::hash<uint64_t>{}(x);
	}
};

} // namespace external

using external::custom;
using external::lookup_custom_tag;

} // namespace ANN

#endif // _ANN_CUSTOM_HPP