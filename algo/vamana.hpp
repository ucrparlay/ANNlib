#ifndef _ANN_ALGO_VAMANA_HPP
#define _ANN_ALGO_VAMANA_HPP

#include <cstdint>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <memory>
#include <functional>
#include <ranges>
#include <fstream>
#include <string>
#include <unordered_map>
#include <iterator>
#include <type_traits>
#include <limits>
#include <thread>
#include "algo/algo.hpp"
#include "map/direct.hpp"
#include "util/debug.hpp"
#include "util/helper.hpp"
#include "util/vec.hpp"
#include "custom/custom.hpp"

namespace ANN::vamana_details{

template<typename Nid>
struct edge : util::conn<Nid>{
	constexpr bool operator<(const edge &rhs) const{
		return this->u<rhs.u;
	}
	constexpr bool operator>(const edge &rhs) const{
		return this->u>rhs.u;
	}
	constexpr bool operator==(const edge &rhs) const{
		return this->u==rhs.u;
	}
	constexpr bool operator!=(const edge &rhs) const{
		return this->u!=rhs.u;
	}

	mutable uint32_t livestamp = 0;
};

} // namespace ANN::vamana_details

template<typename Nid>
struct std::hash<ANN::vamana_details::edge<Nid>>{
	size_t operator()(const ANN::vamana_details::edge<Nid> &e) const noexcept{
		return std::hash<decltype(e.u)>{}(e.u);
	}
};

namespace ANN{

template<class Desc>
class vamana
{
	using cm = custom<typename lookup_custom_tag<Desc>::type>;

	template<typename T>
	using seq = typename cm::seq<T>;

	typedef uint32_t nid_t;
	using point_t = typename Desc::point_t;
	using pid_t = typename point_t::id_t;
	using coord_t = typename point_t::coord_t;
	using md_t = util::vec<seq<float>>;
	using dist_t = typename Desc::dist_t; // TODO: elaborate
	using conn = util::conn<nid_t>;
	using edge = vamana_details::edge<nid_t>;
	using search_control = algo::search_control;
	using prune_control = algo::prune_control;

public:
	struct result_t{
		dist_t dist;
		pid_t pid;
	};
	/*
		Construct from the vectors [begin, end).
		std::iterator_trait<Iter>::value_type ought to be convertible to T
		dim: 		vector dimension
		R: 			max degree
		L:			beam size during the construction
		alpha:		parameter of the heuristic (similar to the one in vamana)
		batch_base: growth rate of the batch size (discarded because of two passes)
	*/
	vamana(uint32_t dim, uint32_t R=50, uint32_t L=75, float alpha=1.0);
	/*
		Construct from the saved model
		getter(i) returns the actual data (convertible to type T) of the vector with id i
	*/

	template<typename Iter>
	void insert(Iter begin, Iter end, float batch_base=2);

	template<typename Iter>
	void erase(Iter begin, Iter end);

	template<class Seq=seq<result_t>>
	Seq search(
		const coord_t &cq, uint32_t k, uint32_t ef, const search_control &ctrl={}
	) const;

private:
	/*
	static seq<edge>&& edge_cast(seq<conn> &&cs){
		return reinterpret_cast<seq<edge>&&>(std::move(cs));
	}
	static const seq<edge>& edge_cast(const seq<conn> &cs){
		return reinterpret_cast<const seq<edge>&>(cs);
	}
	static seq<conn>&& conn_cast(seq<edge> &&es){
		return reinterpret_cast<seq<conn>&&>(std::move(es));
	}
	static const seq<conn>& conn_cast(const seq<edge> &es){
		return reinterpret_cast<const seq<conn>&>(es);
	}
	*/
	seq<edge> edge_cast(const seq<conn> &cs){
		// TODO: use range in C++20
		auto es = util::delayed_seq(cs.size(), [&](size_t i){
			return edge{cs[i], deltick};
		});
		return util::to<seq<edge>>(es);
	}
	seq<conn> conn_cast(const seq<edge> &es){
		return util::to<seq<conn>>(es);
	}

	struct node_t{
		coord_t coord;

		coord_t& get_coord(){
			return coord;
		}
		const coord_t& get_coord() const{
			return coord;
		}
	};

	using graph_t = typename Desc::graph_t<nid_t,node_t,edge>;

	graph_t g;
	map::direct<pid_t,nid_t> id_map;
	uint32_t deltick = 1;

	nid_t ep; // entry point
	md_t medoid;
	uint32_t dim;
	uint32_t R;
	uint32_t L;
	float alpha;

	template<typename Iter>
	void insert_batch_impl(Iter begin, Iter end);

	uint32_t get_deg_bound() const{
		return R;
	}

	auto gen_f_dist(const coord_t &c) const{

		class dist_evaluator{
			std::reference_wrapper<const graph_t> g;
			std::reference_wrapper<const coord_t> c;
			uint32_t dim;
		public:
			dist_evaluator(const graph_t &g, const coord_t &c, uint32_t dim):
				g(g), c(c), dim(dim){
			}
			dist_t operator()(nid_t v) const{
				return Desc::distance(c.get(), g.get().get_node(v)->get_coord(), dim);
			}
			dist_t operator()(nid_t u, nid_t v) const{
				return Desc::distance(
					g.get().get_node(u)->get_coord(),
					g.get().get_node(v)->get_coord(),
					dim
				);
			}
		};

		return dist_evaluator(g, c, dim);
	}
	auto gen_f_dist(nid_t u) const{
		return gen_f_dist(g.get_node(u)->get_coord());
	}

	auto gen_f_nbhs() const{
		return [&](nid_t u){
			auto f = std::views::filter([&](const edge &e){
				auto &ls = e.livestamp;
				if(ls==0) return false;
				if(ls==deltick) return true;
				ls = id_map.contain_nid(e.u)? deltick: 0;
				return ls!=0;
			});
			auto t = std::views::transform([&](const edge &e){return e.u;});

			if constexpr(std::is_reference_v<decltype(g.get_edges(u))>)
				return std::ranges::ref_view(g.get_edges(u)) | f | t;
			else
				return std::ranges::owning_view(g.get_edges(u)) | f | t;
		};
	}

	template<class Op>
	auto calc_degs(Op op) const{
		seq<size_t> degs(cm::num_workers(), 0);
		g.for_each([&](auto p){
			auto &deg = degs[cm::worker_id()];
			deg = op(deg, g.get_edges(p).size());
		});
		return cm::reduce(degs, size_t(0), op);
	}

public:
	size_t num_nodes() const{
		return g.num_nodes();
	}

	size_t num_edges(nid_t u) const{
		return g.get_edges(u).size();
	}
	size_t num_edges() const{
		return calc_degs(std::plus<>{});
	}

	size_t max_deg() const{
		return calc_degs([](size_t x, size_t y){
			return std::max(x, y);
		});
	}
};

template<class Desc>
vamana<Desc>::vamana(uint32_t dim, uint32_t R, uint32_t L, float alpha) :
	dim(dim), R(R), L(L), alpha(alpha)
{
}

template<class Desc>
template<typename Iter>
void vamana<Desc>::insert(Iter begin, Iter end, float batch_base)
{
	static_assert(std::is_same_v<typename std::iterator_traits<Iter>::value_type, point_t>);
	static_assert(std::is_base_of_v<
		std::random_access_iterator_tag, typename std::iterator_traits<Iter>::iterator_category
	>);

	const size_t n = std::distance(begin, end);
	if(n==0) return;

	// std::random_device rd;
	auto perm = cm::random_permutation(n/*, rd()*/);
	auto rand_seq = util::delayed_seq(n, [&](size_t i) -> decltype(auto){
		// return *(begin+perm[i]); // CHECK: restore before release
		return *(begin+i);
	});

	size_t cnt_skip = 0;
	if(g.empty())
	{
		auto init = rand_seq.begin();
		ep = id_map.insert(init->get_id());
		medoid = md_t(util::inner_t{}, init->get_coord(), dim);
		g.add_node(ep, node_t{init->get_coord()});
		cnt_skip = 1;
	}

	size_t batch_begin=0, batch_end=cnt_skip, size_limit=std::max<size_t>(n*0.02,20000);
	float progress = 0.0;
	while(batch_end<n)
	{
		batch_begin = batch_end;
		batch_end = std::min({n, (size_t)std::ceil(batch_begin*batch_base)+1, batch_begin+size_limit});

		util::debug_output("Batch insertion: [%u, %u)\n", batch_begin, batch_end);
		insert_batch_impl(rand_seq.begin()+batch_begin, rand_seq.begin()+batch_end);
		// insert(rand_seq.begin()+batch_begin, rand_seq.begin()+batch_end, false);

		if(batch_end>n*(progress+0.05))
		{
			progress = float(batch_end)/n;
			// fprintf(stderr, "Built: %3.2f%%\n", progress*100);
			// fprintf(stderr, "# visited: %lu\n", cm::reduce(per_visited));
			// fprintf(stderr, "# eval: %lu\n", cm::reduce(per_eval));
			// fprintf(stderr, "size of C: %lu\n", cm::reduce(per_size_C));
			// per_visited.clear();
			// per_eval.clear();
			// per_size_C.clear();
		}
	}

	// fprintf(stderr, "# visited: %lu\n", cm::reduce(per_visited));
	// fprintf(stderr, "# eval: %lu\n", cm::reduce(per_eval));
	// fprintf(stderr, "size of C: %lu\n", cm::reduce(per_size_C));
	// per_visited.clear();
	// per_eval.clear();
	// per_size_C.clear();
}

template<class Desc>
template<typename Iter>
void vamana<Desc>::insert_batch_impl(Iter begin, Iter end)
{
	const size_t size_batch = std::distance(begin,end);
	seq<nid_t> nids(size_batch);

	// per_visited.resize(size_batch);
	// per_eval.resize(size_batch);
	// per_size_C.resize(size_batch);

	// before the insertion, prepare the needed data
	// `nids[i]` is the nid of the node corresponding to the i-th 
	// point to insert in the batch, associated with level[i]
	id_map.insert(util::delayed_seq(size_batch, [&](size_t i){
		return (begin+i)->get_id();
	}));

	cm::parallel_for(0, size_batch, [&](uint32_t i){
		nids[i] = id_map.get_nid((begin+i)->get_id());
	});

	g.add_nodes(util::delayed_seq(size_batch, [&](size_t i){
		// GUARANTEE: begin[*].get_coord is only invoked for assignment once
		return std::pair{nids[i], node_t{(begin+i)->get_coord()}};
	}));

	const auto n_prev = g.num_nodes();
	// TODO: make util::filter compatible with ranges
/*	auto coord_drift = cm::reduce(
		std::ranges::subrange(begin, end) |
		std::views::filter([&](auto &&p){return !!id_map.contain_nid(p.get_id());}) |
		std::views::transform([&](auto &&p){return md_t(util::inner_t{},p.get_coord(),dim);})
	);*/
	auto it_refs = util::delayed_seq(size_batch, [&](size_t i){
		return begin+i;
	});
	auto refs_new = util::filter(it_refs, [&](Iter it){
		return !!id_map.find_nid(it->get_id());
	});
	auto coords_new = util::delayed_seq(
		refs_new.size(),
		[&](size_t i){return md_t(util::inner_t{},refs_new[i]->get_coord(),dim);}
	);
	md_t coord_drift = cm::reduce(coords_new, md_t(dim));

	// below we (re)generate edges incident to nodes in the current batch
	// add adges from the new points
	// TODO: change edge_added to conn_added
	seq<seq<std::pair<nid_t,edge>>> edge_added(size_batch);
	seq<std::pair<nid_t,seq<edge>>> nbh_forward(size_batch);
	cm::parallel_for(0, size_batch, [&](size_t i){
		const nid_t u = nids[i];

		search_control sctrl; // TODO: use designated initializers in C++20
		sctrl.log_per_stat = i;
		seq<conn> res = algo::beamSearch(gen_f_nbhs(), gen_f_dist(u), seq<nid_t>{ep}, L, sctrl);

		prune_control pctrl; // TODO: use designated intializers in C++20
		pctrl.alpha = alpha;
		seq<conn> conn_u = algo::prune_heuristic(
			std::move(res), get_deg_bound(), 
			gen_f_nbhs(), gen_f_dist(u), pctrl
		);
		// record the edge for the backward insertion later
		auto &edge_cur = edge_added[i];
		edge_cur.clear();
		edge_cur.reserve(conn_u.size());
		for(const auto &[d,v] : conn_u)
			edge_cur.emplace_back(v, edge{{d,u}, deltick});

		// store for batch insertion
		nbh_forward[i] = {u, edge_cast(std::move(conn_u))};
	});
	util::debug_output("Adding forward edges\n");
	g.set_edges(std::move(nbh_forward));

	// now we add edges in the other direction
	auto edge_added_flatten = util::flatten(std::move(edge_added));
	auto edge_added_grouped = util::group_by_key(std::move(edge_added_flatten));

	// TODO: use std::remove_cvref in C++20
	using agent_t = std::remove_cv_t<std::remove_reference_t<
		decltype(g.get_edges(nid_t()))
	>>;
	seq<std::pair<nid_t,agent_t>> nbh_backward(edge_added_grouped.size());

	cm::parallel_for(0, edge_added_grouped.size(), [&](size_t j){
		nid_t v = edge_added_grouped[j].first;
		auto &nbh_v_add = edge_added_grouped[j].second;

		auto edge_agent_v = g.get_edges(v);
		auto edge_v = util::to<seq<edge>>(
			std::move(edge_agent_v) |
			std::views::filter([&](const edge &e){
				return id_map.contain_nid(e.u);
			})
		);
		edge_v.insert(edge_v.end(),
			std::make_move_iterator(nbh_v_add.begin()),
			std::make_move_iterator(nbh_v_add.end())
		);

		seq<conn> conn_v = algo::prune_simple(
			conn_cast(std::move(edge_v)),
			get_deg_bound()
		);
		edge_agent_v = edge_cast(std::move(conn_v));
		nbh_backward[j] = {v, std::move(edge_agent_v)};
	});
	util::debug_output("Adding backward edges\n");
	g.set_edges(std::move(nbh_backward));

	// finally, update the entrances
	util::debug_output("Updating entrance\n");
	const auto n_curr = g.num_nodes();
	((medoid*=n_prev)+=coord_drift)/=n_curr;
	util::vec<seq<typename point_t::elem_t>> t(util::inner_t{}, medoid.data(), dim);
	ep = algo::beamSearch(
		gen_f_nbhs(),
		[&](nid_t v){return Desc::distance(
			t.data(), g.get_node(v)->get_coord(), dim);
		},
		seq<nid_t>{ep},
		1
	)[0].u;
}

template<class Desc>
template<typename Iter>
void vamana<Desc>::erase(Iter begin, Iter end)
{
	static_assert(std::is_convertible_v<
		typename std::iterator_traits<Iter>::value_type, pid_t
	>);
	/*
	static_assert(std::is_base_of_v<
		std::random_access_iterator_tag,
		typename std::iterator_traits<Iter>::iterator_category
	>);*/

	auto nids = std::ranges::subrange(begin,end) |
		std::views::transform([&](const pid_t &p){return id_map.get_nid(p);});

	auto r = nids | std::views::transform([&](nid_t u){
		return md_t(util::inner_t{}, g.get_node(u)->get_coord(), dim);
	});
	auto drift = cm::reduce(r);

	((medoid*=g.num_nodes())-=drift);
	g.remove_nodes(nids.begin(), nids.end());

	id_map.erase(begin, end);
	deltick++;

	medoid /= g.num_nodes();
	util::vec<seq<typename point_t::elem_t>> t(util::inner_t{}, medoid.data(), dim);
	ep = algo::beamSearch(
		gen_f_nbhs(),
		[&](nid_t v){return Desc::distance(
			t.data(), g.get_node(v)->get_coord(), dim);
		},
		seq<nid_t>{id_map.front_nid()},
		1
	)[0].u;
}

template<class Desc>
template<class Seq>
Seq vamana<Desc>::search(
	const coord_t &cq, uint32_t k, uint32_t ef, const search_control &ctrl) const
{
	auto nbhs = beamSearch(gen_f_nbhs(), gen_f_dist(cq), seq<nid_t>{ep}, ef, ctrl);

	nbhs = algo::prune_simple(std::move(nbhs), k/*, ctrl*/); // TODO: set ctrl
	cm::sort(nbhs.begin(), nbhs.end());

	using result_t = typename Seq::value_type;
	static_assert(util::is_direct_list_initializable_v<
		result_t, dist_t, pid_t
	>);
	Seq res(nbhs.size());
	cm::parallel_for(0, nbhs.size(), [&](size_t i){
		const auto &nbh = nbhs[i];
		res[i] = result_t{nbh.d, id_map.get_pid(nbh.u)};
	});

	return res;
}

} // namespace ANN

#endif // _ANN_ALGO_VAMANA_HPP
