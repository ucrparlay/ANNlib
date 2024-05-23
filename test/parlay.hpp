#ifndef _ANN_TEST_PARLAY_HPP
#define _ANN_TEST_PARLAY_HPP

#include <utility>
#include <ranges>
#include "custom/undef.hpp"
#include <parlay/parallel.h>
#include <parlay/primitives.h>

namespace ANN::external{

class custom_tag_parlay{};

template<>
class custom<custom_tag_parlay> : public custom<custom_tag_undef>
{
	using base = custom<custom_tag_undef>;

	/*
	template<typename T_, class BinaryOp>
	struct monoid{
		using T = T_;
		T identity;
		BinaryOp op;
		constexpr T f(const T &a, const T &b){
			return op(a, b);
		}
	};
	*/ // TODO: delete

public:
	template<typename T>
	using alloc = parlay::allocator<T>;

	template<typename T>
	using seq = parlay::sequence<T>;

	template<typename F>
	static void parallel_for(
		size_t start, size_t stop, F f,
		long granularity=0, bool conservative=false)
	{
		parlay::parallel_for(
			start, stop, std::move(f), 
			granularity, conservative
		);
	}

	static uint64_t hash64(uint64_t x)
	{
		return parlay::hash64_2(x);
	}

	template<typename Iter, typename Comp=std::less<>>
	static void sort(Iter begin, Iter end, Comp comp={})
	{
		if(std::distance(begin,end)<10000)
			std::sort(begin, end, comp);
		else
			parlay::sort_inplace(parlay::make_slice(begin,end), comp);
	}

	template<class R, // TODO: shorten
		class T=std::remove_reference_t<std::ranges::range_value_t<typename std::remove_reference_t<R>>>,
		class BinaryOp=std::plus<>>
	static auto reduce(R &&range, T init={}, BinaryOp op={})
	{
		return parlay::reduce(std::forward<R>(range),
			parlay::binary_op(std::move(op),std::move(init))
		);
	}

	template<typename Iter,
		class T=std::remove_reference_t<typename std::iterator_traits<Iter>::value_type>,
		class BinaryOp=std::plus<>>
	static auto reduce(Iter begin, Iter end, T init={}, BinaryOp op={})
	{
		return parlay::reduce(parlay::make_slice(begin,end),
			parlay::binary_op(std::move(op),std::move(init))
		);
	}

	static auto worker_id()
	{
		return parlay::worker_id();
	}
	static size_t num_workers(){
		return parlay::num_workers();
	}

	template<template<typename> class TSeq=seq, typename T>
	static TSeq<T> random_permutation(T n)
	{
		auto perm = parlay::random_permutation(n);
		if constexpr(std::is_same_v<TSeq<T>,decltype(perm)>)
			return perm;
		else
			return TSeq<T>(perm.begin(), perm.end());
	}

	template<class Seq>
	static auto pack_index(Seq &&seq)
	{
		return parlay::pack_index(std::forward<Seq>(seq));
	}

	template<class Seq>
	static auto flatten(Seq &&seq)
	{
		return parlay::flatten(std::forward<Seq>(seq));
	}

	template<class Seq>
	static auto group_by_key(Seq &&seq)
	{
		return parlay::group_by_key(std::forward<Seq>(seq));
	}

	template<typename Seq, class F, class L=lookup_custom_tag<>>
	static auto filter(Seq &&seq, F &&f)
	{
		return parlay::filter(std::forward<Seq>(seq), std::forward<F>(f));
	}

	template<typename Iter>
	static Iter max_element(Iter begin, Iter end)
	{
		return parlay::max_element(parlay::make_slice(begin,end));
	}
};

} // namespace ANN::external

#endif // _ANN_TEST_PARLAY_HPP
