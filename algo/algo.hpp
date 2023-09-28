#ifndef _ANN_ALGO_HPP
#define _ANN_ALGO_HPP

#include <cmath>
#include <utility>
#include <algorithm>
#include <functional>
#include <set>
#include <unordered_set>
#include <optional>
#include "ANN.hpp"
#include "custom.hpp"

namespace ANN{
namespace algo{

struct search_control{
	bool verbose_output = false;
	bool skip_search = false;
	float beta = 1;
	std::optional<float> radius;
	std::optional<uint32_t> log_per_stat;
	std::optional<uint32_t> log_dist;
	std::optional<uint32_t> log_size;
	std::optional<uint32_t> indicate_ep;
	std::optional<uint32_t> limit_eval;
};

struct prune_control{
	bool prune_nbh = false;
	bool recycle_pruned = false;
	float alpha = 1;
};

template<class L=lookup_custom_tag<>, class G, class D, class Seq>
auto beamSearch(
	const G &g, D f_dist, const Seq &eps, uint32_t ef, const search_control &ctrl={})
{
	using cm = custom<typename L::type>;
	using nid_t = typename G::nid_t;
	using conn = util::conn<nid_t>;

	const auto n = g.num_nodes();
	const uint32_t bits = ef>2? std::ceil(std::log2(ef))*2-2: 2;
	const uint32_t mask = (1u<<bits)-1;
	Seq visited(mask+1, n+1);
	uint32_t cnt_visited = 0;
	typename cm::seq<conn> workset;
	std::set<conn> cand; // TODO: test dual heaps
	std::unordered_set<nid_t> is_inw; // TODO: test merge instead
	// TODO: get statistics about the merged size
	// TODO: switch to the alternative if exceeding a threshold
	workset.reserve(ef+1);

	for(nid_t pe : eps)
	{
		visited[cm::hash64(pe)&mask] = pe;
		const auto d = f_dist(g.get_node(pe)->get_coord());
		cand.insert({d,pe});
		workset.push_back({d,pe});
		is_inw.insert(pe);
	}
	std::make_heap(workset.begin(), workset.end());

	uint32_t cnt_eval = 0;
	uint32_t limit_eval = ctrl.limit_eval.value_or(n);
	while(cand.size()>0)
	{
		if(cand.begin()->d>workset[0].d*ctrl.beta) break;

		if(++cnt_eval>limit_eval) break;

		cand.erase(cand.begin());
		for(nid_t pv: g.get_edges(cand.begin()->u))
		{
			const auto h_pv = cm::hash64(pv)&mask;
			if(visited[h_pv]==pv) continue;
			visited[h_pv] = pv;
			cnt_visited++;

			const auto d = f_dist(g.get_node(pv)->get_coord());
			if(!(workset.size()<ef||d<workset[0].d)) continue;
			if(is_inw.insert(pv).second) continue;

			cand.insert({d,pv});
			workset.push_back({d,pv});
			std::push_heap(workset.begin(), workset.end());
			if(workset.size()>ef)
			{
				std::pop_heap(workset.begin(), workset.end());
				is_inw.erase(workset.back().u);
				workset.pop_back();
			}
			if(cand.size()>ef)
				cand.erase(std::prev(cand.end()));
		}
	}

	/*
	if(ctrl.log_per_stat)
	{
		const auto qid = *ctrl.log_per_stat;
		per_visited[qid] += cnt_visited;
		per_eval[qid] += C.size()+cnt_eval;
		per_size_C[qid] += cnt_eval;
	}
	*/
	return workset;
}

namespace detail{

template<class T>
struct second_elem{
	static auto helper(T &&t){
		auto [_,y] = t;
		return y;
	}
	using type = decltype(helper(std::declval<T>()));
};

template<class T>
using second_elem_t = typename second_elem<T>::type;

} // namespace detail

template<class L=lookup_custom_tag<>, class Seq>
Seq prune_simple(
	Seq cand, uint32_t size, const prune_control &ctrl={})
{
	(void)ctrl;
	using nid_t = detail::second_elem_t<typename Seq::value_type>;
	using conn = util::conn<nid_t>;
	static_assert(std::is_same_v<typename Seq::value_type,conn>);

	if(cand.size()>size)
	{
		std::nth_element(cand.begin(), cand.begin()+size, cand.end());
		cand.resize(size);
	}
	return cand;
}

template<class L=lookup_custom_tag<>, class Seq, class D, class G>
Seq prune_heuristic(
	Seq &&cand, uint32_t size, D f_dist, G g, const prune_control &ctrl={})
{
	using cm = custom<typename L::type>;
	using nid_t = detail::second_elem_t<typename Seq::value_type>;
	using conn = util::conn<nid_t>;

	Seq workset = std::forward<Seq>(cand);
	/*
	if(ctrl.extend_nbh)
	{
		const auto &g = ctrl.graph;
		std::unordered_set<nid_t> cand_ext;
		for(const conn &c : workset)
		{
			cand_ext.insert(c.u);
			for(nid_t pv : g.get_edges(c.u))
				cand_ext.insert(pv);
		}

		workset.reserve(workset.size()+cand_ext.size());
		for(nid_t pc : cand_ext)
			workset.push_back({f_dist(g.get_node(pc).get_coord()), pc});
		cand_ext.clear();
	}
	*/
	cm::sort(workset.begin(), workset.end());

	Seq res, pruned;
	std::unordered_set<nid_t> nbh;
	for(conn &c : workset)
	{
		const auto d_cu = c.d*ctrl.alpha;

		bool is_pruned = false;
		for(const conn &r : res)
		{
			const auto d_cr = f_dist(
				g.get_node(c.u)->get_coord(),
				g.get_node(r.u)->get_coord()
			);
			if(d_cr<d_cu)
			{
				is_pruned = true;
				break;
			}
		}

		if(!is_pruned && ctrl.prune_nbh)
			is_pruned = nbh.find(c.u)!=nbh.end();

		if(!is_pruned)
		{
			if(ctrl.prune_nbh)
				for(nid_t pv : g.get_edges(c.u))
					nbh.insert(pv);

			res.push_back(std::move(c));
			if(res.size()==size) break;
		}
		else pruned.push_back(std::move(c));
	}

	if(ctrl.recycle_pruned)
	{
		size_t cnt_recycle = std::min(pruned.size(), size-res.size());
		auto split = pruned.begin()+cnt_recycle;
		std::nth_element(pruned.begin(), split, pruned.end());
		res.insert(res.end(),
			std::make_move_iterator(pruned.begin()),
			std::make_move_iterator(split)
		);
	}
	return res;
}

} // namespace algo
} // namespace ANN

#endif // _ANN_ALGO_HPP