#ifndef _ANN_ALGO_HCNNG_HPP
#define _ANN_ALGO_HCNNG_HPP

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
#include <optional>
#include <type_traits>
#include <limits>
#include <thread>
#include "algo/algo.hpp"
#include "algo/vamana.hpp"
#include "map/direct.hpp"
#include "util/debug.hpp"
#include "util/helper.hpp"
#include "util/vec.hpp"
#include "custom/custom.hpp"

namespace ANN::HCNNG_details{

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

} // namespace ANN::HCNNG_details

template<typename Nid>
struct std::hash<ANN::HCNNG_details::edge<Nid>>{
	size_t operator()(const ANN::HCNNG_details::edge<Nid> &e) const noexcept{
		return std::hash<decltype(e.u)>{}(e.u);
	}
};

namespace ANN{

template<class Desc>
class HCNNG
{
	using cm = custom<typename lookup_custom_tag<Desc>::type>;

	template<typename T>
	using seq = typename cm::seq<T>;

	typedef uint32_t nid_t;
	using point_t = typename Desc::point_t;
	using pid_t = typename point_t::id_t;
	using coord_t = typename point_t::coord_t;
	using dist_t = typename Desc::dist_t; // TODO: elaborate
	using conn = util::conn<nid_t>;
	using edge = HCNNG_details::edge<nid_t>;
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
		deg:		the max degree allowed in the final graph
		alpha:		parameter of the heuristic pruning
		num_cl:		the number of clusters
		cl_size:	the max size of a leaf
		mst_deg:	the max degree in each MST
	*/
	template<typename Iter>
	HCNNG(Iter begin, Iter end, uint32_t dim,
		uint32_t deg=40, float alpha=1.0,
		uint32_t num_cl=24, uint32_t cl_size=1000, uint32_t mst_deg=3);
	/*
		Construct from the saved model
		getter(i) returns the actual data (convertible to type T) of the vector with id i
	*/

	template<class Seq=seq<result_t>>
	Seq search(
		const coord_t &cq, uint32_t k, uint32_t ef, const search_control &ctrl={}
	) const;

private:
	struct node_t{
		coord_t coord;

		coord_t& get_coord(){
			return coord;
		}
		const coord_t& get_coord() const{
			return coord;
		}
	};

	struct desc_leaf{
		struct point_t{
			using id_t = uint32_t;
			using elem_t = typename Desc::point_t::elem_t;
			using coord_t = typename Desc::coord_t;

			id_t get_id() const{return id;}
			const coord_t& get_coord() const{return coord;}

			id_t id;
			std::reference_wrapper<const coord_t> coord;
		};

		using dist_t = typename Desc::dist_t;
		static dist_t distance(
			const coord_t &cu, const coord_t &cv, uint32_t dim){
			return Desc::distance(cu, cv, dim);
		}

		template<typename Nid, class Ext, class Edge>
		using graph_t = typename Desc::graph_t<Nid,Ext,Edge>;
	};

	using graph_t = typename Desc::graph_t<nid_t,node_t,edge>;

	graph_t g;
	map::direct<pid_t,nid_t> id_map;

	nid_t ep; // entry point
	uint32_t dim;
	uint32_t deg;
	float alpha;
	uint32_t num_cl;
	uint32_t cl_size;
	uint32_t mst_deg;

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

		return dist_evaluator(g, c, dim);
	}
	auto gen_f_dist(nid_t u) const{
		return gen_f_dist(g.get_node(u)->get_coord());
	}

	auto gen_f_nbhs() const{
		return [&](nid_t u){
			auto t = std::views::transform([&](const edge &e){return e.u;});

			if constexpr(std::is_reference_v<decltype(g.get_edges(u))>)
				return std::ranges::ref_view(g.get_edges(u)) | t;
			else
				return std::ranges::owning_view(g.get_edges(u)) | t;
		};
	}

	void print_stat() const{
		puts("#vertices         edges  avg. deg");
		size_t cnt_vertex = num_nodes();
		size_t cnt_degree = num_edges();
		printf("%14lu %16lu %10.2f\n", 
			cnt_vertex, cnt_degree, float(cnt_degree)/cnt_vertex
		);
	}

	template<class R>
	void build_tree(R ps);
	template<class R, class EA>
	void build_tree(R ps, EA as, std::minstd_rand rng);

	template<class R, class EA>
	void build_leaf(R ps, EA as);

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
template<typename Iter>
HCNNG<Desc>::HCNNG(Iter begin, Iter end, uint32_t dim,
	uint32_t deg, float alpha, uint32_t num_cl, uint32_t cl_size, uint32_t mst_deg) :
	dim(dim), deg(deg), alpha(alpha), num_cl(num_cl), cl_size(cl_size), mst_deg(mst_deg)
{
	const size_t n = std::distance(begin,end);
	seq<nid_t> nids(n);

	// TODO: remove the explicit initialization
	per_visited.resize(n);
	per_eval.resize(n);
	per_size_C.resize(n);

	// before the construction, prepare the needed data
	// `nids[i]` is the nid of the node corresponding to the i-th 
	// point to insert in the batch, associated with level[i]
	id_map.insert(util::delayed_seq(n, [&](size_t i){
		return (begin+i)->get_id();
	}));

	cm::parallel_for(0, n, [&](uint32_t i){
		nids[i] = id_map.get_nid((begin+i)->get_id());
	});

	g.add_nodes(util::delayed_seq(n, [&](size_t i){
		// GUARANTEE: begin[*].get_coord is only invoked for assignment once
		return std::pair{nids[i], node_t{(begin+i)->get_coord()}};
	}));

	auto collect = util::delayed_seq(n, [&](size_t i){
		nid_t u = nids[i];
		return std::pair(u, std::cref(g.get_node(u)->get_coord()));
	});
	auto ps = util::to<seq<typename decltype(collect)::value_type>>(collect);
	for(uint32_t i=0; i<num_cl; ++i)
	{
		printf("Building cluster #%u\n", i);
		build_tree(std::ranges::subrange(ps));
		print_stat();
	}
}

template<class Desc>
template<class R> // value_type = pair<nid_t, reference_wrapper<coord_t>>
void HCNNG<Desc>::build_tree(R ps)
{
	using ea_t = std::remove_cvref_t<decltype(g.get_edges(nid_t()))>;
	const auto n = ps.size();
	seq<std::optional<ea_t>> as(n);
	build_tree(ps, std::ranges::subrange(as), std::minstd_rand(std::random_device()()));
	g.set_edges(util::delayed_seq(n,[&](size_t i){
		return std::pair{ps[i].first, std::move(*as[i])};
	}));
} 

template<class Desc>
template<class R, class EA>
void HCNNG<Desc>::build_tree(R ps, EA as, std::minstd_rand rng)
{
	static_assert(sizeof(nid_t)<=sizeof(decltype(rng)::result_type));
	if(ps.size()<=cl_size)
	{
		build_leaf(ps, as);
		return;
	}

	size_t idx_a = rng()%ps.size();
	size_t idx_b = rng()%(ps.size()-1);
	if(idx_b==idx_a)
		idx_b = ps.size() - 1;
	auto coord_a = ps[idx_a].second;
	auto coord_b = ps[idx_b].second;

	auto splitter = cm::partition(ps, [&](const auto &p){
		auto dist_a = Desc::distance(p.second, coord_a, dim);
		auto dist_b = Desc::distance(p.second, coord_b, dim);
		return dist_a < dist_b;
	});
	using std::ranges::subrange;
	using std::ranges::begin;
	using std::ranges::end;
	subrange ps_l(begin(ps),splitter), ps_r(splitter,end(ps));
	const auto size_l = ps_l.size();
	auto as_l = as | std::views::take(size_l);
	auto as_r = as | std::views::drop(size_l);

	auto rng_l = rng;
	rng.discard(size_l*2);
	auto &rng_r = rng;
	cm::par_do(
		[&](){build_tree(ps_l, as_l, rng_l);},
		[&](){build_tree(ps_r, as_r, rng_r);}
	);
}

template<class Desc>
template<class R, class EA>
void HCNNG<Desc>::build_leaf(R ps, EA as)
{
	// TODO: parameterize the numbers
	// TODO: use map::trivial
	const auto n = ps.size();
	vamana<desc_leaf> idx(dim, 8, 12, 0.75);
	auto ss = util::delayed_seq(n, [&](uint32_t i){
		return typename desc_leaf::point_t{i, ps[i].second};
	});
	idx.insert(ss.begin(), ss.end());

	seq<seq<edge>> nbhs(n);
	// seq<seq<std::pair<nid_t,edge>>> edge_added(size_batch);
	for(size_t i=0; i<n; ++i) // use parallel for-loop instead?
	{
		// TODO: parameterize the numbers
		using iedge = HCNNG_details::edge<uint32_t>;
		auto res = idx.template search<seq<iedge>>(
			ss[i].get_coord(), mst_deg, mst_deg*1.8
		);
		for(const auto &[d,j] : res)
		{
			nbhs[i].push_back({d, ps[j].first});
			nbhs[j].push_back({d, ps[i].first});
		}
	};

	// TODO: optimize adding edges in small sizes
	for(size_t i=0; i<n; ++i)
	{
		nid_t u = ps[i].first;
		auto &nbh_add = nbhs[i];

		auto ea = g.get_edges(u);
		auto edges = util::to<seq<edge>>(std::move(ea));
		edges.insert(edges.end(),
			std::make_move_iterator(nbh_add.begin()),
			std::make_move_iterator(nbh_add.end())
		);

		prune_control pctrl{.alpha=alpha};
		seq<conn> conns = algo::prune_heuristic(
			conn_cast(std::move(edges)), deg,
			gen_f_nbhs(), gen_f_dist(u), pctrl
		);

		ea = edge_cast(std::move(conns));
		as[i] = std::move(ea);
	}
}

template<class Desc>
template<class Seq>
Seq HCNNG<Desc>::search(
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

#endif // _ANN_ALGO_HCNNG_HPP
