#ifndef _ANN_ALGO_HNSW_HPP
#define _ANN_ALGO_HNSW_HPP

#include <cstdint>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <memory>
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

namespace ANN{

template<class Desc>
class HNSW
{
	using point_t = typename Desc::point_t;
	using pid_t = typename point_t::id_t;
	using coord_t = typename point_t::coord_t;
	using dist_t = typename Desc::dist_t; // TODO: elaborate
	typedef uint32_t nid_t;
	using conn = util::conn<nid_t>;
	using cm = custom<typename lookup_custom_tag<Desc>::type>;
	template<typename T>
	using seq = typename cm::seq<T>;
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
		m_l: 		control the # of levels (larger m_l leads to more layer)
		m: 			max degree
		efc:		beam size during the construction
		alpha:		parameter of the heuristic (similar to the one in vamana)
		batch_base: growth rate of the batch size (discarded because of two passes)
	*/
	template<typename Iter>
	HNSW(
		Iter begin, Iter end,
		uint32_t dim, float m_l=0.4, 
		uint32_t m=100, uint32_t efc=50, 
		float alpha=1.0, float batch_base=2
	);
	/*
		Construct from the saved model
		getter(i) returns the actual data (convertible to type T) of the vector with id i
	*/

	template<class Seq=seq<result_t>>
	Seq search(
		const coord_t &cq, uint32_t k, uint32_t ef, const search_control &ctrl={}
	) const;

private:
	struct node_lite{
		seq<dist_t> dist_nbh;

		coord_t& get_coord(); // not in use
		const coord_t& get_coord() const; // not in use
	};
	struct node_fat{
		uint32_t level;
		coord_t coord;
		seq<dist_t> dist_nbh;

		coord_t& get_coord(){
			return coord;
		}
		const coord_t& get_coord() const{
			return coord;
		}
	};

	using seq_nid = seq<nid_t>;

	using graph_fat = typename Desc::graph_t<nid_t,node_fat>;
	using graph_lite = typename Desc::graph_aux<nid_t,node_lite>;

	// the bottom layer 0 is stored in `layer_b` with all needed data (level, coordinate, etc)
	// the upper layers 1..lev_ep are stored in `layer_u[]` 
	// where each node associates with the one representing itself at the bottom layer
	// std::vector<graph::adj_map<nid_t,node_lite>> layer_u;
	std::vector<graph_lite> layer_u;
	// graph::adj_seq<nid_t,node_fat> layer_b;
	graph_fat layer_b;
	map::direct<pid_t,nid_t> id_map;

	seq_nid entrance; // To init
	uint32_t dim;
	float m_l;
	uint32_t m;
	uint32_t efc;
	float alpha;
	uint32_t n;

	template<typename Iter>
	void insert_batch_impl(Iter begin, Iter end);

	template<class Seq=seq_nid>
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
		return [&](nid_t u) -> decltype(auto){return g.get_edges(u);};
	}

	// TODO: eliminate redundant code by deducing 'this' in C++23
	seq<dist_t>& get_dist_nbh(uint32_t l, nid_t u){
		return l==0? layer_b.get_node(u)->dist_nbh: layer_u[l].get_node(u)->dist_nbh;
	}
	const seq<dist_t>& get_dist_nbh(uint32_t l, nid_t u) const{
		return l==0? layer_b.get_node(u)->dist_nbh: layer_u[l].get_node(u)->dist_nbh;
	}

public:
	uint32_t get_height(nid_t u) const
	{
		return layer_b.get_node(u)->level;
	}

	uint32_t get_height() const
	{
		return get_height(entrance[0]);
	}

	size_t cnt_degree(uint32_t l, nid_t u) const{
		uint32_t level_u = get_height(u);
		return level_u<l? 0:
			l==0? layer_b.get_degree(u):
			layer_u[l].get_degree(u);
	}
	size_t cnt_degree(uint32_t l) const
	{
		auto cnt_each = util::delayed_seq(n, [&](size_t i){
			return cnt_degree(l, nid_t(i));
		});
		return cm::reduce(cnt_each);
	}

	size_t cnt_vertex(uint32_t l) const
	{
		auto cnt_each = util::delayed_seq(n, [&](size_t i){
			return get_height(nid_t(i))<l? 0: 1;
		});
		return cm::reduce(cnt_each);
	}

	size_t get_degree_max(uint32_t l) const
	{
		auto cnt_each = util::delayed_seq(n, [&](size_t i){
			return cnt_degree(l, nid_t(i));
		});
		return cm::reduce(cnt_each, 0, [](size_t x, size_t y){
			return std::max(x, y);
		});
	}
};

template<class Desc>
template<typename Iter>
HNSW<Desc>::HNSW(Iter begin, Iter end, uint32_t dim_, 
	float m_l_, uint32_t m_, uint32_t efc, float alpha, float batch_base
):
	dim(dim_), m_l(m_l_), m(m_), efc(efc), alpha(alpha), n(std::distance(begin,end))
{
	static_assert(std::is_same_v<typename std::iterator_traits<Iter>::value_type, point_t>);
	static_assert(std::is_base_of_v<
		std::random_access_iterator_tag, typename std::iterator_traits<Iter>::iterator_category
	>);

	if(n==0) return; // TODO: remove n

	// std::random_device rd;
	auto perm = cm::random_permutation(n/*, rd()*/);
	auto rand_seq = util::delayed_seq(n, [&](size_t i) -> decltype(auto){
		return *(begin+perm[i]);
	});

	const auto level_ep = gen_level();
	const nid_t ep_init = id_map.insert(rand_seq.begin()->get_id());
	layer_b.add_node(ep_init, node_fat{level_ep,rand_seq.begin()->get_coord(),{}});
	if(level_ep>0)
	{
		layer_u.resize(level_ep+1);
		for(uint32_t l=level_ep; l>0; --l)
			layer_u[l].add_node(ep_init, node_lite{});
	}
	entrance.push_back(ep_init);

	uint32_t batch_begin=0, batch_end=1, size_limit=n*0.02;
	float progress = 0.0;
	while(batch_end<n)
	{
		batch_begin = batch_end;
		batch_end = std::min({n, (uint32_t)std::ceil(batch_begin*batch_base)+1, batch_begin+size_limit});

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

	#if 0
		for(const auto *pu : node_pool)
		{
			fprintf(stderr, "[%u] (%.2f,%.2f)\n", U::get_id(get_node(pu).data), get_node(pu).data[0], get_node(pu).data[1]);
			for(int32_t l=pu->level; l>=0; --l)
			{
				fprintf(stderr, "\tlv. %d:", l);
				for(const auto *k : pu->neighbors[l])
					fprintf(stderr, " %u", U::get_id(get_node(k).data));
				fputs("\n", stderr);
			}
		}
	#endif
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

	const uint32_t level_ep = layer_b.get_node(entrance[0])->level;
	const uint32_t level_max = level[0];

	layer_b.add_nodes(util::delayed_seq(size_batch, [&](size_t i){
		// GUARANTEE: only invoke once
		return std::pair{nids[i], node_fat{level[i],(begin+i)->get_coord(),{}}};
	}));
	if(level_max>level_ep)
		layer_u.resize(level_max+1);
	// note the end of range where level==0 is not yet in `pos_split'
	for(size_t j=1; j<pos_split.size(); ++j)
	{
		const uint32_t l_hi = level[pos_split[j-1]];
		const uint32_t l_lo = level[pos_split[j]];
		for(uint32_t l=l_hi; l>l_lo; --l) // TODO: why isn't it a problem in CPAM
		{
			util::debug_output(
				"== insert [%u, %u) to layer[%u]\n", 
				begin->get_id(), (begin+pos_split[j])->get_id(), l
			);
			layer_u[l].add_nodes(
				util::delayed_seq(pos_split[j], [&](size_t i){
					return std::pair{nids[i], node_lite{}};
				})
			);
		}
	}

	pos_split.push_back(size_batch); // complete the range of level 0
	util::debug_output("results of pos_split: ");
	for(size_t j : pos_split)
		util::debug_output("%lu ", j);
	util::debug_output("\n");

	// below we (re)generate edges incident to nodes in the current batch
	// in the top-to-bottom manner

	// first, query the nearest point as the entry point for each node
	cm::parallel_for(0, size_batch, [&](size_t i){
		nid_t u = nids[i];
		eps[i] = search_layer_to(layer_b.get_node(u)->get_coord(), 1, level[i]);
	});
	util::debug_output("Finish searching entrances\n");

	// then we process them layer by layer (`l': current layer)
	size_t j = 0;
	for(int32_t l=std::min(level_ep,level_max); l>=0; --l) // TODO: fix the type
	{
		util::debug_output("Looking for neighbors on lev. %d\n", l);

		while(pos_split[j+1]<size_batch && level[pos_split[j+1]]>=l)
			j++;
		util::debug_output("j=%lu l=%d\n", j, l);

		auto set_edges = [&](auto &&...args){
			if(l==0)
				layer_b.set_edges(std::forward<decltype(args)>(args)...);
			else
				layer_u[l].set_edges(std::forward<decltype(args)>(args)...);
		};

		// nodes indexed within [0, pos_split[j+1]) have their levels>='l'
		seq<seq<std::pair<nid_t,conn>>> edge_added(pos_split[j+1]);
		seq<std::pair<nid_t,seq_nid>> nbh_forward(pos_split[j+1]);
		cm::parallel_for(0, pos_split[j+1], [&](size_t i){
			const nid_t u = nids[i];

			auto &eps_u = eps[i];
			auto search_layer = [&](const auto &g) -> decltype(auto){
				search_control ctrl; // TODO: use designated initializers in C++20
				ctrl.log_per_stat = i;
				return algo::beamSearch(gen_f_nbhs(g), gen_f_dist(u), eps_u, efc, ctrl);
			};
			auto res = l==0? search_layer(layer_b): search_layer(layer_u[l]);

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
			auto nbh_u = l==0? prune(layer_b): prune(layer_u[l]);
			// record the edge for the backward insertion later
			auto &edge_cur = edge_added[i];
			edge_cur.clear();
			edge_cur.reserve(nbh_u.size());
			for(const auto &[d,v] : nbh_u)
				edge_cur.emplace_back(v, conn{d,u});

			seq_nid edge_u(nbh_u.size());
			auto &dist_u = get_dist_nbh(l,u);
			dist_u.resize(nbh_u.size());
			cm::parallel_for(0, nbh_u.size(), [&](size_t i){
				const auto &[d,u] = nbh_u[i];
				edge_u[i] = u;
				dist_u[i] = d;
			});
			// store for batch insertion
			nbh_forward[i] = {u, std::move(edge_u)};
		});
		util::debug_output("Adding forward edges\n");
		set_edges(std::move(nbh_forward));

		// now we add edges in the other direction
		auto edge_added_flatten = util::flatten(edge_added);
		auto edge_added_grouped = util::group_by_key(edge_added_flatten);
		seq<std::pair<nid_t,seq_nid>> nbh_backward(edge_added_grouped.size());

		cm::parallel_for(0, edge_added_grouped.size(), [&](size_t j){
			nid_t v = edge_added_grouped[j].first;
			auto &nbh_v_add = edge_added_grouped[j].second;

			auto edge_v_old = l==0? layer_b.pop_edges(v): layer_u[l].pop_edges(v);
			auto &dist_v_old = get_dist_nbh(l,v);
			/*
			auto dist_v_old_d = util::delayed_seq(edge_v_old.size(), [&](size_t i){
				return Desc::distance(
					layer_b.get_node(v)->get_coord(),
					layer_b.get_node(edge_v_old[i])->get_coord(),
					dim
				);
			});
			seq<dist_t> dist_v_old(dist_v_old_d.begin(), dist_v_old_d.end());
			*/
			auto nbh_v_old_d = util::zip<seq<conn>>(dist_v_old, edge_v_old);
			seq<conn> nbh_v_old(nbh_v_old_d.begin(), nbh_v_old_d.end());

			const auto m_s = get_deg_bound(l);
			nbh_v_old.insert(nbh_v_old.end(),
				std::make_move_iterator(nbh_v_add.begin()),
				std::make_move_iterator(nbh_v_add.end())
			);

			auto nbh_v = algo::prune_simple(std::move(nbh_v_old),m_s);

			seq_nid edge_v(nbh_v.size());
			dist_v_old.clear();
			auto &dist_v = dist_v_old;
			dist_v.resize(nbh_v.size());
			cm::parallel_for(0, nbh_v.size(), [&](size_t i){
				const auto &[d,v] = nbh_v[i];
				edge_v[i] = v;
				dist_v[i] = d;
			});
			nbh_backward[j] = {v, std::move(edge_v)};
		});
		util::debug_output("Adding backward edges\n");
		set_edges(std::move(nbh_backward));
	} // for-loop l

	// finally, update the entrances
	util::debug_output("Updating entrance\n");
	if(level_max>level_ep)
	{
		util::debug_output("Prompt the ep_level to %u\n", level_max);
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

	seq_nid eps;
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
