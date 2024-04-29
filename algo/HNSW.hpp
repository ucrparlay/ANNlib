#ifndef _ANN_ALGO_HNSW_HPP
#define _ANN_ALGO_HNSW_HPP

#include <cstdint>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <memory>
#include <functional>
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
#include "custom/custom.hpp"

namespace ANN::HNSW_details{

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
};

} // namespace ANN::HNSW_details

template<typename Nid>
struct std::hash<ANN::HNSW_details::edge<Nid>>{
	size_t operator()(const ANN::HNSW_details::edge<Nid> &e) const noexcept{
		return std::hash<decltype(e.u)>{}(e.u);
	}
};

namespace ANN{

template<class Desc>
class HNSW
{
	using cm = custom<typename lookup_custom_tag<Desc>::type>;

	typedef uint32_t nid_t;
	using point_t = typename Desc::point_t;
	using pid_t = typename point_t::id_t;
	using coord_t = typename point_t::coord_t;
	using dist_t = typename Desc::dist_t; // TODO: elaborate
	using conn = util::conn<nid_t>;
	using edge = HNSW_details::edge<nid_t>;
	using search_control = algo::search_control;
	using prune_control = algo::prune_control;

	template<typename T>
	using seq = typename cm::seq<T>;

public:
	struct result_t{
		dist_t dist;
		pid_t pid;
	};
	/*
		Construct from the vectors [begin, end).
		std::iterator_trait<Iter>::value_type ought to be convertible to T
		dim: 		vector dimension
		m_l: 		control the # of levels (larger m_l leads to more layer)
		m: 			max degree
		efc:		beam size during the construction
		alpha:		parameter of the heuristic (similar to the one in vamana)
		batch_base: growth rate of the batch size (discarded because of two passes)
	*/
	HNSW(
		uint32_t dim, float m_l=0.4, 
		uint32_t m=100, uint32_t efc=50, 
		float alpha=1.0
	);
	/*
		Construct from the saved model
		getter(i) returns the actual data (convertible to type T) of the vector with id i
	*/

	template<typename Iter>
	void insert(Iter begin, Iter end, float batch_base=2);

	template<class Seq=seq<result_t>>
	Seq search(
		const coord_t &cq, uint32_t k, uint32_t ef, const search_control &ctrl={}
	) const;

private:
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

	struct node_lite{
		coord_t& get_coord(); // not in use
		const coord_t& get_coord() const; // not in use
	};

	struct node_fat{
		uint32_t level;
		coord_t coord;

		coord_t& get_coord(){
			return coord;
		}
		const coord_t& get_coord() const{
			return coord;
		}
	};

	using graph_fat = typename Desc::graph_t<nid_t,node_fat,edge>;
	using graph_lite = typename Desc::graph_aux<nid_t,node_lite,edge>;

	// the bottom layer 0 is stored in `layer_b` with all needed data (level, coordinate, etc)
	// the upper layers 1..lev_ep are stored in `layer_u[]` 
	// where each node associates with the one representing itself at the bottom layer
	// std::vector<graph::adj_map<nid_t,node_lite>> layer_u;
	std::vector<graph_lite> layer_u;
	// graph::adj_seq<nid_t,node_fat> layer_b;
	graph_fat layer_b;
	map::direct<pid_t,nid_t> id_map;

	seq<nid_t> entrance; // To init
	uint32_t dim;
	float m_l;
	uint32_t m;
	uint32_t efc;
	float alpha;

	template<typename Iter>
	void insert_batch_impl(Iter begin, Iter end);

	template<class Seq=seq<nid_t>>
	Seq search_layer_to(
		const coord_t &cq, uint32_t ef, uint32_t l_stop, const search_control &ctrl={}
	) const;

	uint32_t get_deg_bound(uint32_t level) const{
		return level==0? m*2: m;
	}

	uint32_t gen_level() const{
		// static thread_local int32_t anchor;
		// uint32_t esp;
		// asm volatile("movl %0, %%esp":"=a"(esp));
		// static thread_local std::hash<std::thread::id> h;
		// static thread_local std::mt19937 gen{h(std::this_thread::get_id())};
		static thread_local std::mt19937 gen{cm::worker_id()};
		static thread_local std::uniform_real_distribution<> dis(std::numeric_limits<float>::min(), 1.0);
		const uint32_t res = uint32_t(-log(dis(gen))*m_l);
		return res;
	}

	auto gen_f_dist(const coord_t &c) const{

		class dist_evaluator{
			std::reference_wrapper<const graph_fat> g;
			std::reference_wrapper<const coord_t> c;
			uint32_t dim;
		public:
			dist_evaluator(const graph_fat &g, const coord_t &c, uint32_t dim):
				g(g), c(c), dim(dim){
			}
			dist_t operator()(nid_t v) const{
				return Desc::distance(c, g.get().get_node(v)->get_coord(), dim);
			}
			dist_t operator()(nid_t u, nid_t v) const{
				return Desc::distance(
					g.get().get_node(u)->get_coord(),
					g.get().get_node(v)->get_coord(),
					dim
				);
			}
		};

		return dist_evaluator(layer_b, c, dim);
	}
	auto gen_f_dist(nid_t u) const{
		return gen_f_dist(layer_b.get_node(u)->get_coord());
	}

	template<class G>
	auto gen_f_nbhs(const G &g) const{
		return [&](nid_t u) -> decltype(auto){
			// TODO: use std::views::transform in C++20 / util::tranformed_view
			// TODO: define the return type as a view, and use auto to receive it
			if constexpr(std::is_reference_v<decltype(g.get_edges(u))>)
			{
				const auto &edges = g.get_edges(u);
				return util::delayed_seq(edges.size(), [&](size_t i){
					return edges[i].u;
				});
			}
			else
			{
				auto edges = g.get_edges(u);
				return util::delayed_seq(edges.size(), [=](size_t i){
					return edges[i].u;
				});
			}	
		};
	}

	template<class Op>
	auto calc_degs(uint32_t l, Op op) const{
		auto impl = [&](const auto &g){
			seq<size_t> degs(cm::num_workers(), 0);
			g.for_each([&](auto p){
				auto &deg = degs[cm::worker_id()];
				deg = op(deg, g.get_edges(p).size());
			});
			return cm::reduce(degs, size_t(0), op);
		};
		return l==0? impl(layer_b): impl(layer_u[l]);
	}

public:
	uint32_t get_height(nid_t u) const{
		return layer_b.get_node(u)->level;
	}
	uint32_t get_height() const{
		return get_height(entrance[0]); // TODO: fix the issue when being empty
	}

	size_t num_nodes(uint32_t l) const{
		return l==0? layer_b.num_nodes(): layer_u[l].num_nodes();
	}

	size_t num_edges(uint32_t l, nid_t u) const{
		auto num_edges_impl = [u](const auto &g){
			return g.get_edges(u).size();
		};
		return num_edges_impl(l==0? layer_b: layer_u[l]);
	}
	size_t num_edges(uint32_t l) const{
		return calc_degs(l, std::plus<>{});
	}

	size_t max_deg(uint32_t l) const{
		return calc_degs(l, [](size_t x, size_t y){
			return std::max(x, y);
		});
	}

	auto collect_refs(uint32_t l) const{
		auto impl = [](const auto &g){
			using Map = std::unordered_map<uint64_t,uint64_t>;
			seq<Map> ref_cnt(cm::num_workers());
			g.for_each_raw([&](const auto &e, auto *ptr, auto r){
				(void)r;
				auto &counter = ref_cnt[cm::worker_id()];
				// uint64_t id = (uintptr_t(ptr)<<32) | std::get<0>(e);
				uint64_t id = uintptr_t(ptr);
				const auto &[key,val] = e;
				const auto *ptr_el = val.get_edges_raw().data();
				// uint64_t id = uintptr_t(ptr_el);
				const auto size_el = val.get_edges().size();
				counter[id] = (uintptr_t(ptr_el)<<16) | size_el;
			});
			Map all_cnts;
			for(auto &counter : ref_cnt)
				all_cnts.merge(std::move(counter));
			return all_cnts;
		};
		return l==0? impl(layer_b): impl(layer_u[l]);
	}

	decltype(auto) get_nbhs(nid_t u) const{
		return gen_f_nbhs(layer_b)(u);
	}
};

template<class Desc>
HNSW<Desc>::HNSW(
	uint32_t dim_, float m_l_, uint32_t m_, uint32_t efc, float alpha
) :
	dim(dim_), m_l(m_l_), m(m_), efc(efc), alpha(alpha)
{
}

template<class Desc>
template<typename Iter>
void HNSW<Desc>::insert(Iter begin, Iter end, float batch_base)
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
		return *(begin+perm[i]);
	});

	size_t cnt_skip = 0;
	if(layer_b.empty())
	{
		const auto level_ep = gen_level();
		const nid_t ep_init = id_map.insert(rand_seq.begin()->get_id());
		layer_b.add_node(ep_init, node_fat{level_ep,rand_seq.begin()->get_coord()});
		if(level_ep>0)
		{
			layer_u.resize(level_ep+1);
			for(uint32_t l=level_ep; l>0; --l)
				layer_u[l].add_node(ep_init, node_lite{});
		}
		entrance.push_back(ep_init);
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
			fprintf(stderr, "Built: %3.2f%%\n", progress*100);
			fprintf(stderr, "# visited: %lu\n", cm::reduce(per_visited));
			fprintf(stderr, "# eval: %lu\n", cm::reduce(per_eval));
			fprintf(stderr, "size of C: %lu\n", cm::reduce(per_size_C));
			per_visited.clear();
			per_eval.clear();
			per_size_C.clear();
		}
	}

	fprintf(stderr, "# visited: %lu\n", cm::reduce(per_visited));
	fprintf(stderr, "# eval: %lu\n", cm::reduce(per_eval));
	fprintf(stderr, "size of C: %lu\n", cm::reduce(per_size_C));
	per_visited.clear();
	per_eval.clear();
	per_size_C.clear();
}

template<class Desc>
template<typename Iter>
void HNSW<Desc>::insert_batch_impl(Iter begin, Iter end)
{
	const size_t size_batch = std::distance(begin,end);
	seq<uint32_t> level(size_batch);
	seq<nid_t> nids(size_batch);
	seq<seq<nid_t>> eps(size_batch);

	per_visited.resize(size_batch);
	per_eval.resize(size_batch);
	per_size_C.resize(size_batch);

	// before the insertion, prepare the needed data
	// `nids[i]` is the nid of the node corresponding to the i-th 
	// point to insert in the batch, associated with level[i]
	id_map.insert(util::delayed_seq(size_batch, [&](size_t i){
		return (begin+i)->get_id();
	}));

	cm::parallel_for(0, size_batch, [&](uint32_t i){
		const nid_t u = id_map.get_nid((begin+i)->get_id());
		nids[i] = u;
		level[i] = gen_level();
	});

	// the points compose nodes with other attributes (e.g., level)
	// initially, we insert the nodes into graphs
	// to do so in batches, we sort levels assigned to nodes,
	// which is equivalent to grouping nodes by level
	// since the order of points does not matter
	cm::sort(level.begin(), level.end(), std::greater<uint32_t>{});
	auto pos_split = util::pack_index(util::delayed_seq(size_batch, [&](size_t i){
		return i==0 || level[i-1]!=level[i];
	}));

	// with the level info, query the nearest nbhs as entry points for each node
	cm::parallel_for(0, size_batch, [&](size_t i){
		eps[i] = search_layer_to((begin+i)->get_coord(), 1, level[i]);
	});
	util::debug_output("Finish searching entrances\n");

	// next, add the nodes themselves into the graphs
	const uint32_t level_ep = layer_b.get_node(entrance[0])->level;
	const uint32_t level_max = level[0];
	if(level_max>level_ep)
		layer_u.resize(level_max+1);

	auto add_to_upper = [&](uint32_t l, size_t pos_end){
		util::debug_output(
			"== insert [%u, %u) to layer[%u]\n", 
			begin->get_id(), (begin+pos_end)->get_id(), l
		);
		layer_u[l].add_nodes(
			util::delayed_seq(pos_end, [&](size_t i){
				return std::pair{nids[i], node_lite{}};
			})
		);
	};
	// note the end of range where level==0 is not yet in `pos_split'
	// so any level[pos_split[j]] must be valid
	for(size_t j=1; j<pos_split.size(); ++j)
	{
		const uint32_t l_hi = level[pos_split[j-1]];
		const uint32_t l_lo = level[pos_split[j]];
		for(uint32_t l=l_hi; l>l_lo; --l)
			add_to_upper(l, pos_split[j]);
	}
	for(uint32_t l=level[pos_split.back()]; l>0; --l)
		add_to_upper(l, size_batch);

	layer_b.add_nodes(util::delayed_seq(size_batch, [&](size_t i){
		// GUARANTEE: begin[*].get_coord is only invoked for assignment once
		return std::pair{nids[i], node_fat{level[i],(begin+i)->get_coord()}};
	}));

	pos_split.push_back(size_batch); // complete the range of level 0
	util::debug_output("results of pos_split: ");
	for(size_t j : pos_split)
		util::debug_output("%lu ", j);
	util::debug_output("\n");

	// below we (re)generate edges incident to nodes in the current batch
	// we process them layer by layer (`l': current layer) in a top-to-bottom manner
	size_t j = 0;
	for(int32_t l=std::min(level_ep,level_max); l>=0; --l) // TODO: fix the type
	{
		util::debug_output("Looking for neighbors on lev. %d\n", l);

		// nodes indexed within [0, pos_end) have their levels>='l'
		size_t pos_end;
		while(pos_end=pos_split[j+1],
			pos_end<size_batch && level[pos_end]>=l)
			j++;
		util::debug_output("j=%lu l=%d\n", j, l);

		auto set_edges = [&](auto &&...args){
			if(l==0)
				layer_b.set_edges(std::forward<decltype(args)>(args)...);
			else
				layer_u[l].set_edges(std::forward<decltype(args)>(args)...);
		};
		// add adges from the new points
		seq<seq<std::pair<nid_t,edge>>> edge_added(pos_end);
		seq<std::pair<nid_t,seq<edge>>> nbh_forward(pos_end);
		cm::parallel_for(0, pos_end, [&](size_t i){
			const nid_t u = nids[i];

			auto &eps_u = eps[i];
			auto search_layer = [&](const auto &g) -> decltype(auto){
				search_control ctrl; // TODO: use designated initializers in C++20
				ctrl.log_per_stat = i;
				return algo::beamSearch(gen_f_nbhs(g), gen_f_dist(u), eps_u, efc, ctrl);
			};
			seq<conn> res = l==0? search_layer(layer_b): search_layer(layer_u[l]);

			// prepare the entrance points for the next layer
			eps_u.clear();
			eps_u.reserve(res.size());
			for(const auto &c : res)
				eps_u.push_back(c.u);

			auto prune = [&](const auto &g) -> decltype(auto){
				prune_control ctrl; // TODO: use designated intializers in C++20
				ctrl.alpha = alpha;
				return algo::prune_heuristic(
					std::move(res), get_deg_bound(l), 
					gen_f_nbhs(g), gen_f_dist(u), ctrl
				);
			};
			seq<conn> conn_u = l==0? prune(layer_b): prune(layer_u[l]);
			// record the edge for the backward insertion later
			auto &edge_cur = edge_added[i];
			edge_cur.clear();
			edge_cur.reserve(conn_u.size());
			for(const auto &[d,v] : conn_u)
				edge_cur.emplace_back(v, edge{d,u});

			// store for batch insertion
			nbh_forward[i] = {u, edge_cast(std::move(conn_u))};
		});
		util::debug_output("Adding forward edges\n");
		set_edges(std::move(nbh_forward));

		// now we add edges in the other direction
		auto edge_added_flatten = util::flatten(std::move(edge_added));
		auto edge_added_grouped = util::group_by_key(std::move(edge_added_flatten));

		auto add_rev_nbhs = [&](auto &g){
			// TODO: use std::remove_cvref in C++20
			using agent_t = std::remove_cv_t<std::remove_reference_t<
				decltype(g.get_edges(nid_t()))
			>>;
			seq<std::pair<nid_t,agent_t>> nbh_backward(edge_added_grouped.size());

			cm::parallel_for(0, edge_added_grouped.size(), [&](size_t j){
				nid_t v = edge_added_grouped[j].first;
				auto &nbh_v_add = edge_added_grouped[j].second;

				auto edge_agent_v = g.get_edges(v);
				auto edge_v = util::to<seq<edge>>(std::move(edge_agent_v));
				edge_v.insert(edge_v.end(),
					std::make_move_iterator(nbh_v_add.begin()),
					std::make_move_iterator(nbh_v_add.end())
				);

				seq<conn> conn_v = algo::prune_simple(
					conn_cast(std::move(edge_v)),
					get_deg_bound(l)
				);
				edge_agent_v = edge_cast(conn_v);
				nbh_backward[j] = {v, std::move(edge_agent_v)};
			});
			set_edges(std::move(nbh_backward));
		};
		l==0? add_rev_nbhs(layer_b): add_rev_nbhs(layer_u[l]);
		util::debug_output("Adding backward edges\n");
	} // for-loop l

	// finally, update the entrances
	util::debug_output("Updating entrance\n");
	if(level_max>level_ep)
	{
		util::debug_output("Promote the ep_level to %u\n", level_max);
		entrance.clear();
	}

	if(level_max>=level_ep)
	{
		util::debug_output("Insert %lu nodes to the top level\n", pos_split[1]);
		entrance.insert(entrance.end(), nids.begin(), nids.begin()+pos_split[1]);
	}
}

template<class Desc>
template<class Seq>
Seq HNSW<Desc>::search_layer_to(
	const coord_t &cq, uint32_t ef, uint32_t l_stop, const search_control &ctrl) const
{
	auto eps = entrance;
	for(uint32_t l=layer_b.get_node(eps[0])->level; l>l_stop; --l)
	{
		search_control c{};
		c.log_per_stat = ctrl.log_per_stat; // whether count dist calculations at all layers
		// c.limit_eval = ctrl.limit_eval; // whether apply the limit to all layers
		const auto W = beamSearch(gen_f_nbhs(layer_u[l]), gen_f_dist(cq), eps, ef, c);
		eps.clear();
		eps.push_back(W[0].u);
		/*
		while(!W.empty())
		{
			eps.push_back(W.top().u);
			W.pop();
		}
		*/
	}
	return eps;
}

template<class Desc>
template<class Seq>
Seq HNSW<Desc>::search(
	const coord_t &cq, uint32_t k, uint32_t ef, const search_control &ctrl) const
{
	/*
	const auto wid = cm::worker_id();
	total_range_candidate[wid] = 0;
	total_visited[wid] = 0;
	total_eval[wid] = 0;
	total_size_C[wid] = 0;
	if(ctrl.log_per_stat)
	{
		const auto qid = *ctrl.log_per_stat;
		per_visited[qid] = 0;
		per_eval[qid] = 0;
		per_size_C[qid] = 0;
	}
	*/

	seq<nid_t> eps;
	if(ctrl.indicate_ep)
		eps.push_back(*ctrl.indicate_ep);
	else
		eps = search_layer_to(cq, 1, 0, ctrl);
	auto nbhs = beamSearch(gen_f_nbhs(layer_b), gen_f_dist(cq), eps, ef, ctrl);

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

#endif // _ANN_ALGO_HNSW_HPP
