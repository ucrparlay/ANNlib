#ifndef _ANN_CUSTOM_UNDEF_HPP
#define _ANN_CUSTOM_UNDEF_HPP

#include <cstdint>
#include <vector>
#include <algorithm>
#include <memory>
#include <random>
#include <numeric>
// #include <execution>
#include <utility>
#include <ranges>
#include <functional>
#include "custom.hpp"
#include "util/seq.hpp"

namespace ANN::external{

class custom_tag_undef;

template<typename, typename>
struct lookup_custom_tag_impl{
	using type = custom_tag_undef;
};

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
		// std::for_each(std::execution::par, a.begin(), a.end(), f);
		std::for_each(a.begin(), a.end(), f);
	}

	static uint64_t hash64(uint64_t x)
	{
		return std::hash<uint64_t>{}(x);
	}

	template<typename T>
	static seq<T> random_permutation(T n, unsigned int seed=1206)
	{
		seq<T> s(n);
		std::iota(s.begin(), s.end(), 0);
		std::ranges::shuffle(s, std::mt19937(seed));
		return s;
	}

	template<typename Iter, class Comp=std::less<>>
	static void sort(Iter begin, Iter end, Comp comp={})
	{
		std::sort(begin, end, comp);
	}

	template<typename R, class F, class L=lookup_custom_tag<>>
	static auto filter(const R &r, F &&f)
	{
		// TODO: fix the type
		using T = std::ranges::range_value_t<R>;
		auto d = std::views::filter(r, std::forward<F>(f));
		return seq<T>(d.begin(), d.end());
	}

	template<typename Iter,
		class T=std::remove_reference_t<typename std::iterator_traits<Iter>::value_type>,
		class BinaryOp=std::plus<>>
	static auto reduce(Iter begin, Iter end, T init={}, BinaryOp op={})
	{
		return std::reduce(begin, end, init, op);
	}

	template<class R, // TODO: shorten
		class T=std::remove_reference_t<std::ranges::range_value_t<typename std::remove_reference_t<R>>>,
		class BinaryOp=std::plus<>>
	static auto reduce(const R &r, T init={}, BinaryOp op={})
	{
		return reduce(std::ranges::begin(r), std::ranges::end(r), init, op);
	}

	template<class R>
	static auto flatten(const R &r)
	{
		using S = std::ranges::range_value_t<R>;
		S res;
		for(const auto &e : r)
			res.insert(res.end(), e.begin(), e.end());
		return res;
	}

	template<class R>
	static auto group_by_key(R r)
	{
		using E = std::ranges::range_value_t<R>;
		using K = std::tuple_element_t<0, E>;
		using V = std::tuple_element_t<1, E>;
		using S = seq<std::pair<K,seq<V>>>;

		sort(r.begin(), r.end(), [](const E &x, const E &y){
			return std::get<0>(x) < std::get<0>(y);
		});

		auto keys = std::views::keys(r);
		auto values = std::views::values(r);

		seq<size_t> split;
		split.push_back(0);
		size_t n = std::ranges::size(r);
		// TODO: use std::views::enumerate in C++23
		for(size_t i=1; i<n; ++i)
		{
			if(keys[split.back()]!=keys[i])
				split.push_back(i);
		}
		split.push_back(n);

		S res;
		for(size_t i=1; i<split.size(); ++i)
			res.push_back({
				keys[split[i-1]],
				seq<V>(
					values.begin()+split[i-1],
					values.begin()+split[i]
				)
			});
		return res;
	}
};

} // namespace external

#endif // _ANN_CUSTOM_UNDEF_HPP
