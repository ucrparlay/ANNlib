#ifndef _ANN_TEST_ASPEN_HPP
#define _ANN_TEST_ASPEN_HPP

#include <memory>
#include <utility>
#include <type_traits>
#include "util/util.hpp"
#include "util/seq.hpp"
#include "custom/custom.hpp"
#include "aspen.h"

template<class VGraph>
class graph_aspen
{
	using graph_t = typename VGraph::graph_t;
	using vtx_t = typename graph_t::vertex;
	using ext_t = typename graph_t::ext_t;
	using vinfo_t = typename graph_t::vinfo;
	using edge_tree = typename graph_t::edge_tree;

	using cm = ANN::custom<typename ANN::lookup_custom_tag<VGraph>::type>;

public:
	using nid_t = decltype(vtx_t::id);

private:
	template<bool IsConst>
	struct ptr_base{
	protected:
		using attr_t = std::tuple_element_t<1, typename vinfo_t::_base>;

	public:
		using ptr_t = std::conditional_t<IsConst, const attr_t*, attr_t*>;
		using ref_t = std::conditional_t<IsConst, const attr_t&, attr_t&>;

		ref_t operator*() const{
			return const_cast<attr_t&>(std::get<1>(vtx.edges.base()));
		}
		ptr_t operator->() const{
			return &operator*();
		}

	protected:
		using V = std::conditional_t<IsConst, const vtx_t, vtx_t>;
		V vtx;

		ptr_base(const V &vtx) : vtx(vtx){
		}
		ptr_base(V &&vtx) : vtx(std::move(vtx)){
		}

		friend class graph_aspen;
	};

public:
	struct node_ptr : ptr_base<false>{
		using ptr_base<false>::ptr_base;
	};

	struct node_cptr : ptr_base<true>{
		using ptr_base<true>::ptr_base;
		node_cptr(const node_ptr &other) :
			ptr_base<true>(other.vtx){
		}
		node_cptr(node_ptr &&other) :
			ptr_base<true>(std::move(other.vtx)){
		}
	};

private:
	class edgeview{
		using nbh_t = typename graph_t::neighbors;
		nbh_t nbhs;

	public:
		edgeview(const nbh_t &nbhs) : nbhs(nbhs){
		}
		edgeview(nbh_t &&nbhs) : nbhs(std::move(nbhs)){
		}
		template<class F>
		void iter_each(F &&f) /*const*/{
			auto g = [&](nid_t /*u*/, nid_t v, .../*weight*/){
				f(v);
				return true;
			};
			nbhs.foreach_cond(g); // TODO: unify the name
		}
		template<class F>
		void for_each(F &&f) /*const*/{
			auto g = [&](nid_t /*u*/, nid_t v, .../*weight*/){
				f(v);
				return true;
			};
			// TODO: unify the name
			nbhs.map_cond(g, []{return true;});
		}
	};

	edgeview get_edges_impl(node_cptr p, ANN::util::dummy<edgeview>) const{
		return p.vtx.out_neighbors();
	}
	template<class Range>
	const Range get_edges_impl(node_cptr p, ANN::util::dummy<Range>) const{
		// assert(0); // TODO: remove assert(0)
		edgeview edges = get_edges_impl(p, ANN::util::dummy<edgeview>{});

		Range res;
		edges.iter_each([&](nid_t v){
			res.push_back(v); // NOTICE: adapt to .insert()
		});
		return res;
	}

	template<class T>
	node_ptr add_node_impl(nid_t u, T &&ext){
		auto &g = snapshot.graph;
		g.insert_vertex_inplace(u, {nullptr,ext});
		return get_node(u);
	}

public:
	graph_aspen() :
		vg(std::make_shared<VGraph>()),
		snapshot(vg->acquire_version())
	{
	}
	~graph_aspen()
	{
		vg->release_version(std::move(snapshot));
	}

	graph_aspen(graph_aspen&&) = default;
	graph_aspen& operator=(graph_aspen&&) = default;

	graph_aspen(const graph_aspen &other) :
		vg(other.vg),
		snapshot(vg->acquire_version(other.snapshot.timestamp))
	{
	}
	graph_aspen& operator=(const graph_aspen &other)
	{
		snapshot = other.vg->acquire_version(other.snapshot.timestamp);
		vg = other.vg;
		return *this;
	}

	node_ptr get_node(nid_t u){
		auto &g = snapshot.graph;
		return g.get_vertex(u); // TODO: unify the name
	}
	node_cptr get_node(nid_t u) const{
		const auto &g = snapshot.graph;
		return g.get_vertex(u); // TODO: unify the name
	}

	template<class Range=edgeview>
	Range get_edges(node_cptr p) const{
		return get_edges_impl(p, ANN::util::dummy<Range>{}); 
	}
	template<class Range=edgeview>
	decltype(auto) get_edges(nid_t u) const{
		return get_edges<Range>(get_node(u));
	}

	template<class Seq>
	void set_edges(Seq &&ps){
		using vertex_entry_t = typename graph_t::vertex_entry::entry_t;
		auto vs_delayed = ANN::util::delayed_seq(ps.size(), [&](size_t i){
			auto &&[u,es] = ps[i];

			using weight_t = typename graph_t::edge_entry::val_t;
			using edge_entry_t = typename graph_t::edge_entry::entry_t;
			auto ews_delayed = ANN::util::delayed_seq(es.size(), [&](size_t i){
				return edge_entry_t(es[i], weight_t{});
			});

			using ews_t = typename cm::seq<edge_entry_t>;
			ews_t ews(ews_delayed.begin(), ews_delayed.end());

			edge_tree tree(ews.begin(), ews.end());
			auto *root = tree.root;
			tree.root = nullptr;
			return vertex_entry_t(u, vinfo_t(root));
		});
		using vs_t = typename cm::seq<vertex_entry_t>;
		vs_t vs(vs_delayed.begin(), vs_delayed.end());

		class comb{
		public:
			vinfo_t operator()(const vinfo_t &cur, const vinfo_t &inc) const{
				edge_tree t;
				t.root = cur;
				return vinfo_t(std::get<0>(inc), std::get<1>(cur));
			}
			vinfo_t operator()(vinfo_t &&cur, vinfo_t &&inc) const{
				edge_tree t;
				t.root = cur;
				return vinfo_t(std::get<0>(inc), std::get<1>(std::move(cur)));
			}
		};

		auto &g = snapshot.graph;
		g.insert_vertices_batch(vs.size(), vs.data(), comb());
	}

	auto pop_edges(node_cptr p){
		return get_edges<typename cm::seq<nid_t>>(p);
	}
	auto pop_edges(nid_t nid){
		return pop_edges(get_node(nid));
	}

	node_ptr add_node(nid_t nid, const ext_t &ext){
		return add_node_impl(nid, ext);
	}
	node_ptr add_node(nid_t nid, ext_t &&ext){
		return add_node_impl(nid, std::move(ext));
	}
	template<class Seq>
	void add_nodes(Seq &&ns){
		using vertex_entry_t = typename graph_t::vertex_entry::entry_t;
		auto vs_delayed = ANN::util::delayed_seq(ns.size(), [&](size_t i){
			auto &&[nid,ext] = ns[i];
			return vertex_entry_t(nid, vinfo_t(
				nullptr, std::forward<decltype(ext)>(ext)
			));
		});
		using vs_t = typename cm::seq<vertex_entry_t>;
		vs_t vs(vs_delayed.begin(), vs_delayed.end());

		auto &g = snapshot.graph;
		g.insert_vertices_batch(vs.size(), vs.data());
	}

	size_t get_degree(node_cptr p) const{
		// TODO: use `ANN::util::size`
		size_t cnt = 0;
		get_edges(p).iter_each([&](...){cnt++;});
		return cnt;
	}
	size_t get_degree(nid_t u) const{
		return get_degree(get_node(u));
	}

private:
	std::shared_ptr<VGraph> vg;
	typename VGraph::version snapshot;
};

#endif // _ANN_TEST_ASPEN_HPP
