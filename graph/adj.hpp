#ifndef _ANN_GRAPH_ADJVEC_HPP
#define _ANN_GRAPH_ADJVEC_HPP

#include <unordered_map>
#include <cassert>
#include <iterator>
#include <utility>
#include <type_traits>
#include "graph.hpp"
#include "custom/custom.hpp"
#include "util/util.hpp"
#include "util/intrin.hpp"
#include "util/seq.hpp"
#include "util/debug.hpp"
using ANN::util::debug_output;

namespace ANN::graph{

template<
	class Nid, class Ext, class Edge,
	template<typename,typename> class TCtrV, 
	template<typename> class TCtrE>
class adj_base : base
{
public:
	using nid_t = Nid;

protected:
	using cm = custom<typename lookup_custom_tag<Nid>::type>;
	using edgelist = TCtrE<Edge>;
	struct node_t : Ext{
		node_t() = default;
		node_t(const Ext &ext) : Ext(ext){
		}
		node_t(Ext &&ext) : Ext(std::move(ext)){
		}
		using Ext::operator=;

		edgelist neighbors;
	};
	using nodelist = TCtrV<nid_t,node_t>;

private:
	template<bool IsConst>
	struct ptr_base{
		using ptr_t = const Ext*;
		using ref_t = const Ext&;

		ref_t operator*() const{
			return *raw;
		}
		ptr_t operator->() const{
			return raw;
		}

	protected:
		using data_t = std::conditional_t<IsConst, const node_t*, node_t*>;
		data_t raw;
		ptr_base(data_t raw) : raw(raw){
		}

		friend class adj_base;
	};

public:
	struct node_ptr : ptr_base<false>{
		using ptr_base<false>::ptr_base;
	};

	struct node_cptr : ptr_base<true>{
		using ptr_base<true>::ptr_base;
		node_cptr(const node_ptr &other) :
			ptr_base<true>(other.raw){
		}
	};

protected:
	static node_ptr gen_node_ptr(node_t *p){
		return node_ptr(p);
	}
	static node_cptr gen_node_cptr(const node_t *p){
		return node_cptr(p);
	}

	template<class T>
	node_ptr set_node_impl(nid_t nid, T &&ext){
		node_ptr p = get_node(nid);
		*(p.raw) = std::forward<T>(ext);
		return p;
	}

	const edgelist& get_edges_impl(node_cptr p, util::dummy<edgelist>) const{
		return p.raw->neighbors;
	}
	template<class Seq>
	const Seq get_edges_impl(node_cptr p, util::dummy<Seq>) const{
		assert(0); // TODO: remove assert(0)
		const auto &origin = get_edges_impl(p, util::dummy<edgelist>{});
		return Seq(origin.begin(), origin.end());
	}

	edgelist pop_edges_impl(node_ptr p, util::dummy<edgelist>){
		edgelist &nbhs = p.raw->neighbors;
		edgelist edges = std::move(nbhs);
		nbhs.clear();
		return edges;
	}
	template<class Seq>
	Seq pop_edges_impl(node_ptr p, util::dummy<Seq>){
		assert(0);
		edgelist &nbhs = p.raw->neighbors;
		// TODO: use `Seq(std::from_range, nbhs)` in C++23
		Seq edges(
			std::make_move_iterator(nbhs.begin()),
			std::make_move_iterator(nbhs.end())
		);
		nbhs.clear();
		return edges;
	}

public:
	node_ptr get_node(nid_t nid){
		return &nodes[nid];
	}
	node_cptr get_node(nid_t nid) const{
		return &nodes[nid];
	}

	node_ptr set_node(nid_t nid, const Ext &ext){
		return set_node_impl(nid, ext);
	}
	node_ptr set_node(nid_t nid, Ext &&ext){
		return set_node_impl(nid, std::move(ext));
	}

	template<typename Seq=edgelist>
	decltype(auto) get_edges(node_cptr p) const{
		return get_edges_impl(p, util::dummy<Seq>{});
	}
	template<class Seq=edgelist>
	decltype(auto) get_edges(nid_t nid) const{
		return get_edges<Seq>(get_node(nid));
	}

	template<typename Seq=edgelist>
	Seq pop_edges(node_ptr p){
		return pop_edges_impl(p, util::dummy<Seq>{});
	}
	template<class Seq=edgelist>
	Seq pop_edges(nid_t nid){
		return pop_edges<Seq>(get_node(nid));
	}

	template<class Seq>
	void set_edges(node_ptr p, Seq&& es){
		if constexpr(std::is_same_v<std::remove_reference_t<Seq>,edgelist>){
			p.raw->neighbors = std::forward<Seq>(es);
		}
		else{
			assert(0);
			p.raw->neighbors = edgelist(es.begin(),es.end());
		}
	}
	template<class Iter>
	void set_edges(Iter begin, Iter end){
		// TODO: forward to `set_edges(Seq&&)` using `subrange` in C++20
		const auto n = std::distance(begin, end);
		cm::parallel_for(0, n, [&](size_t i){
			auto &&[nid,es] = *(begin+i);
			get_node(nid).raw->neighbors = std::forward<decltype(es)>(es);
		});
	}
	template<class Seq>
	void set_edges(Seq&& ps){
		if constexpr(std::is_rvalue_reference_v<Seq&&>)
		{
			// TODO: use `util::for_each`
			set_edges(
				std::make_move_iterator(ps.begin()),
				std::make_move_iterator(ps.end())
			);
		}
		else set_edges(ps.begin(), ps.end());
	}

	template<class Seq>
	void set_edges(nid_t nid, Seq&& es){
		set_edges(get_node(nid), std::forward<Seq>(es));
	}

	size_t num_nodes() const{
		return nodes.size();
	}
	bool empty() const{
		return nodes.empty();
	}

protected:
	nodelist nodes;
};

namespace detail{

template<typename T, class L=lookup_custom_tag<>>
struct seq_default_impl : custom<typename L::type>::seq<T>{
	using _base = typename custom<typename L::type>::template seq<T>;
	using _base::_base;
	using _base::operator=;
};

template<typename T>
using seq_default = seq_default_impl<T>;

template<typename Key, typename T>
using map_seq = seq_default_impl<T>;

template<typename Key, typename T>
struct map_default : std::unordered_map<Key,T>{
	using _base = std::unordered_map<Key,T>;
	using _base::unordered_map;
	using _base::operator=;
	using _base::operator[];

	const T& operator[](const Key &key) const{
		auto it = _base::find(key);
		// TODO: match `Key' type
		assert(it!=_base::end() || (debug_output("key=%u\n",key),false));
		return it->second;
	}
};

} // namespace detail

template<
	class Nid, class Ext, class Edge=Nid,
	template<typename,typename> class TMapV=detail::map_seq,
	template<typename> class TSeqE=detail::seq_default>
class adj_seq : public adj_base<Nid,Ext,Edge,TMapV,TSeqE>
{
	static_assert(std::is_default_constructible_v<Ext>);

	using _base = adj_base<Nid,Ext,Edge,TMapV,TSeqE>;

	using typename _base::cm;
	using _base::nodes;
	using _base::set_node_impl;

public:
	using typename _base::nid_t;
	using typename _base::node_ptr;
	using _base::gen_node_ptr;
	using _base::gen_node_cptr;
	// using _base::get_node;

private:
	template<class T>
	node_ptr add_node_impl(nid_t nid, T &&ext){
		if(nodes.size()<nid+1)
			nodes.resize(nid+1);
		return set_node_impl(nid, std::forward<T>(ext));
	}

public:
	auto get_node(nid_t nid){
		assert(nid<nodes.size());
		return _base::get_node(nid);
	}
	auto get_node(nid_t nid) const{
		assert(nid<nodes.size());
		return _base::get_node(nid);
	}
	node_ptr add_node(nid_t nid, const Ext &ext){
		return add_node_impl(nid, ext);
	}
	node_ptr add_node(nid_t nid, Ext &&ext){
		return add_node_impl(nid, std::move(ext));
	}
	template<class Iter>
	void add_nodes(Iter begin, Iter end){
		// TODO: forward to `add_nodes(Seq&&)` using `subrange` in C++20
		const auto n = std::distance(begin, end);
		auto nids = util::delayed_seq(n, [&](size_t i){
			return (nid_t)std::get<0>(*(begin+i));
		});
		nid_t nid_max = *cm::max_element(nids.begin(), nids.end());
		if(nodes.size()<nid_max+1)
			nodes.resize(nid_max+1);

		cm::parallel_for(0, n, [&](size_t i){
			// TODO: avoid using nodes
			nodes[nids[i]] = std::get<1>(*(begin+i));
		});
	}
	template<class Seq>
	void add_nodes(Seq&& ns){
		if constexpr(std::is_rvalue_reference_v<Seq&&>)
		{
			// TODO: use `util::for_each`
			add_nodes(
				std::make_move_iterator(ns.begin()),
				std::make_move_iterator(ns.end())
			);
		}
		else add_nodes(ns.begin(), ns.end());
	}

	template<class Iter>
	void remove_nodes(Iter begin, Iter end){
		const auto n = std::distance(begin, end);
		cm::parallel_for(0, n, [&](size_t i){
			nodes[i] = typename decltype(nodes)::value_type{};
		});
	}

	template<class F>
	void iter_each(F &&f) const{
		for(const auto &u : nodes)
			f(gen_node_cptr(&u));
	}
	template<class F>
	void iter_each(F &&f){
		for(auto &u : nodes)
			f(gen_node_ptr(&u));
	}
	// TODO: eliminate redundant code by deducing 'this' in C++23
	template<class F>
	void for_each(F &&f) const{
		cm::parallel_for(0, nodes.size(),
			[&,it=nodes.begin()](size_t i){
				f(gen_node_cptr(&it[i]));
			}
		);
	}
	template<class F>
	void for_each(F &&f){
		cm::parallel_for(0, nodes.size(),
			[&,it=nodes.begin()](size_t i){
				f(gen_node_ptr(&it[i]));
			}
		);
	}
};

template<
	class Nid, class Ext, class Edge=Nid,
	template<typename,typename> class TMapV=detail::map_default,
	template<typename> class TSeqE=detail::seq_default>
class adj_map : public adj_base<Nid,Ext,Edge,TMapV,TSeqE>
{
	using _base = adj_base<Nid,Ext,Edge,TMapV,TSeqE>;
	using typename _base::node_t;
	using _base::nodes;
	using _base::set_node_impl;

public:
	using typename _base::nid_t;
	using typename _base::node_ptr;
	using _base::gen_node_ptr;
	using _base::gen_node_cptr;
	using _base::get_node;

private:
	template<class T>
	node_ptr add_node_impl(nid_t nid, T &&ext){
		return set_node_impl(nid, std::forward<T>(ext));
	}

public:
	node_ptr add_node(nid_t nid, const Ext &ext){
		return add_node_impl(nid, ext);
	}
	node_ptr add_node(nid_t nid, Ext &&ext){
		return add_node_impl(nid, std::move(ext));
	}
	template<class Iter>
	void add_nodes(Iter begin, Iter end){
		// TODO: forward to `add_nodes(Seq&&)` using `subrange` in C++20
		nodes.insert(begin, end);
	}
	template<class Seq>
	void add_nodes(Seq&& ns){
		if constexpr(std::is_rvalue_reference_v<Seq&&>)
		{
			// TODO: use `util::for_each`
			add_nodes(
				std::make_move_iterator(ns.begin()),
				std::make_move_iterator(ns.end())
			);
		}
		else add_nodes(ns.begin(), ns.end());
	}

	template<class Iter>
	void remove_nodes(Iter begin, Iter end){
		for(auto it=begin; it!=end; ++it)
			nodes.erase(*it);
	}

	// TODO: eliminate redundant code by deducing 'this' in C++23
	template<class F>
	void iter_each(F &&f) const{
		util::iter_each(nodes, [&](const auto &p){
			f(gen_node_cptr(&p.second));
		});
	}
	template<class F>
	void iter_each(F &&f){
		util::iter_each(nodes, [&](auto &p){
			f(gen_node_ptr(&p.second));
		});
	}
	template<class F>
	void for_each(F &&f) const{
		util::for_each(nodes, [&](const auto &p){
			f(gen_node_cptr(&p.second));
		});
	}
	template<class F>
	void for_each(F &&f){
		util::for_each(nodes, [&](auto &p){
			f(gen_node_ptr(&p.second));
		});
	}
};

} // namespace ANN

#endif // _ANN_GRAPH_ADJVEC_HPP
