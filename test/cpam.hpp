#ifndef _ANN_TEST_ASPEN_HPP
#define _ANN_TEST_ASPEN_HPP

#include <cstdint>
#include <tuple>
#include <unordered_set>
#include <memory>
#include <functional>
#include <utility>
#include <type_traits>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include "util/util.hpp"
#include "util/iter.hpp"
#include "util/seq.hpp" // TODO: remove once switch to C++20
#include "custom/custom.hpp"
#include "pam/pam.h"
#include "cpam/cpam.h"

struct inner_t{};

// Wrap a (fancy) pointer type UPtr to meet the requirements of A::pointer
// where A is an Allocator following the C++ named requirements
template<class UPtr>
class ptr_adapter : 
	public UPtr,
	public ANN::util::enable_rand_iter<
		ptr_adapter<UPtr>, size_t, ANN::util::empty>
{
	using base_ptr = UPtr;
	using base_iter = ANN::util::enable_rand_iter<
		ptr_adapter<UPtr>, size_t, ANN::util::empty
	>;
	using T = typename UPtr::element_type;

public:
	typedef T value_type;
	typedef T* pointer;
	typedef T& reference;

	ptr_adapter() : base_ptr(), base_iter(0){
	}
	ptr_adapter(const ptr_adapter&) = default;
	ptr_adapter(ptr_adapter&&) = default;
	template<typename ...Args>
	ptr_adapter(inner_t, size_t offset, Args &&...args) :
		base_ptr(std::forward<Args>(args)...),
		base_iter(offset){
	}
	template<typename A, typename ...Args, typename=std::enable_if_t<!std::is_same_v<
		std::remove_cv_t<std::remove_reference_t<A>>,
		ptr_adapter<typename std::pointer_traits<UPtr>::template rebind<std::remove_cv_t<T>>>
	>>>
	ptr_adapter(A &&a, Args &&...args) : 
		ptr_adapter(inner_t{}, size_t(0), std::forward<A>(a), std::forward<Args>(args)...){
	}
	ptr_adapter& operator=(const ptr_adapter&) = default;
	ptr_adapter& operator=(ptr_adapter&&) = default;
	/*
	ptr_adapter(UPtr p) : 
		base_ptr(std::move(p)),
		base_iter(0){
	}
	*/

	pointer operator->() const{
		return base_ptr::operator->()+base_iter::get_pos();
	}
	reference operator*() const{
		return *operator->();
	}
	operator T*() const{
		return operator->();
	}

	// TODO: use the default comparison in C++20
	// bool operator==(const ptr_adapter &rhs) const = default;
	bool operator==(const ptr_adapter &rhs) const{
		return static_cast<const base_ptr&>(*this) ==
			static_cast<const base_ptr&>(rhs) &&
				static_cast<const base_iter&>(*this) ==
			static_cast<const base_iter&>(rhs);
	}
	bool operator!=(const ptr_adapter &rhs) const{
		return !operator==(rhs);
	}
	bool operator==(const nullptr_t &null) const{
		return base_iter::get_pos()==0 &&
			static_cast<const base_ptr&>(*this)==null;
	}
	bool operator!=(const nullptr_t &null) const{
		return !operator==(null);
	}

	template<typename U=T, class CUPtr=std::enable_if_t<
		!std::is_const_v<U>,
		typename std::pointer_traits<UPtr>::template rebind<const U>
	>>
	operator ptr_adapter<CUPtr>() const{
		return {inner_t{}, base_iter::get_pos(), static_cast<const base_ptr&>(*this)};
	}
};

template<class Alloc>
class shared_allocator : public Alloc{
	using at = std::allocator_traits<Alloc>;

	static_assert(std::is_pointer_v<typename at::pointer>);
	using sz_t = typename at::size_type;
	using val_t = typename at::value_type;

	Alloc& base(){
		return static_cast<Alloc&>(*this);
	}
	const Alloc& base() const{
		return static_cast<const Alloc&>(*this);
	}

public:
	using pointer = ptr_adapter<std::shared_ptr<val_t>>;
	using const_pointer = ptr_adapter<std::shared_ptr<const val_t>>;
	using Alloc::Alloc;

	template<typename ...Args>
	pointer allocate(sz_t n, Args ...args){
		return pointer(
			Alloc::allocate(n, std::forward<Args>(args)...),
			[=](auto *p){this->Alloc::deallocate(p,n);},
			base()
		);
	}

	void deallocate(pointer &p, sz_t /*n*/){
		p.reset();
	}
	void deallocate(pointer &&/*p*/, sz_t /*n*/){
		// we do nothing on p as it is destroyed after the call
	}

	template<typename U>
	struct rebind{
		using other = shared_allocator<typename at::template rebind_alloc<U>>;
	};

	bool operator==(const shared_allocator &rhs) const{
		return base()==rhs.base();
	}
	bool operator!=(const shared_allocator &rhs) const{
		return base()!=rhs.base();
	}
};


template<typename T>
class shared_vector
{
	using cm = ANN::custom<typename ANN::lookup_custom_tag<T>::type>;
	using alloc_t = shared_allocator<typename cm::alloc<T>>;
	using ptr_t = typename alloc_t::pointer;
	ptr_t storage;
	size_t len, cap;
	alloc_t alloc;
	static const constexpr float over_resv = 1.6;

	template<typename V>
	void push_back_impl(V &&value){
		if(len<cap){
			new(end()) T(std::forward<V>(value));
			len++;
			return;
		}
		size_t cap_new = (len+1)*over_resv;
		ptr_t s_new = alloc.allocate(cap_new);
		T *p = s_new.operator->();
		for(auto it=begin(); it!=end(); ++it, ++p)
			new(p) T(*it);
		new(p) T(std::forward<V>(value));
		len++;
		if(storage!=nullptr)
			alloc.deallocate(storage, cap);
		storage = std::move(s_new);
		cap = cap_new;
	}
public:
	typedef ptr_t pointer;
	typedef T value_type;
	typedef T* iterator;
	typedef const T* const_iterator;
	typedef T& reference;
	typedef const T& const_reference;
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;

	shared_vector() : len(0), cap(0){
	}
	template<typename Iter>
	shared_vector(Iter start, Iter last) : 
		len(std::distance(start, last)), cap(len*over_resv)
	{
		if(len==0) return;

		storage = alloc.allocate(cap);
		T *p = storage.operator->();
		for(; start!=last; ++start, ++p)
			new(p) T(*start);
	}
	shared_vector(shared_vector&&) noexcept = default;
	shared_vector(const shared_vector&) = default;
	shared_vector& operator=(shared_vector&&) noexcept = default;
	shared_vector& operator=(const shared_vector&) = default;

	T* data() noexcept{
		return storage.operator->();
	}
	const T* data() const noexcept{
		return storage.operator->();
	}
	size_type size() const noexcept{
		return len;
	}
	size_type capacity() const noexcept{
		return cap;
	}
	iterator begin() noexcept{
		return data();
	}
	const_iterator begin() const noexcept{
		return data();
	}
	iterator end() noexcept{
		return data()+len;
	}
	const_iterator end() const noexcept{
		return data()+len;
	}
	reference operator[](size_type pos){
		return data()[pos];
	}
	const_reference operator[](size_type pos) const{
		return data()[pos];
	}
	void push_back(const T &value){
		push_back_impl(value);
	}
	void push_back(T &&value){
		push_back_impl(std::move(value));
	}
	void clear(){
		if(storage!=nullptr)
			alloc.deallocate(storage, cap);
		len = cap = 0;
	}

	~shared_vector(){
		clear();
	}
};


template<typename Nid, class Ext, class Edge=Nid>
class graph_cpam : ANN::graph::base
{
public:
	using nid_t = Nid;
	using ext_t = Ext;

private:
	using cm = ANN::custom<typename ANN::lookup_custom_tag<Nid>::type>;
	// using edgelist = typename cm::seq<Edge>;
	// TODO: make parlay::sequence support Allocator::pointer
	// using edgelist = std::vector<Edge, shared_allocator<typename cm::alloc<Edge>>>;
	// using edgelist = std::vector<Edge>;
	using edgelist = shared_vector<Edge>;

	// vinfo_* are designed to be reference-counted so that
	// copying it over multiple versions is efficient
	// Besides, such design potentially allows nodemap::find
	// in the original CPAM code to safely return a copy of val_t 
	// by using ANN::util::materialized_ptr
	template<typename T>
	class shared_wrapper : public std::shared_ptr<T>{
	private:
		using base = std::shared_ptr<T>;

	public:
		constexpr shared_wrapper() noexcept = default;

		template<typename ...Args>
		shared_wrapper(Args &&...args) : 
			base(std::allocate_shared<T>(
				typename cm::alloc<T>(),
				std::forward<Args>(args)...
			))
		{}
	};


	// TODO: inherit from view in C++20
	
	class const_edge_agent{
		edgelist el;

		const_edge_agent() = default;
		const_edge_agent(const edgelist &other) : 
			el(other){
		}
		const_edge_agent(edgelist &&other) :
			el(std::move(other)){
		}

		edgelist& get_ref() &{
			return el;
		}
		edgelist&& get_ref() &&{
			return std::move(el);
		}
		const edgelist& get_cref() const{
			return el;
		}
		friend graph_cpam;
	public:
		typedef typename edgelist::value_type value_type;

		decltype(auto) operator[](size_t i) const{
			return get_cref()[i];
		}
		auto begin() const{
			return get_cref().begin();
		}
		auto end() const{
			return get_cref().end();
		}
		auto size() const{
			return get_cref().size();
		}
	};
	struct edge_agent : const_edge_agent{
		using const_edge_agent::const_edge_agent;

		template<class R>
		edge_agent& operator=(R &&r){
			// std::sort(r.begin(), r.end());
			edgelist &es = this->get_ref();
			// TODO: handle the case of deletion
			thread_local std::unordered_set<Edge> curr;
			curr.clear();
			for(const Edge &u : es)
				curr.insert(u);

			size_t len_r = 0;
			for(const Edge &c : r)
				if(curr.find(c)==curr.end())
					len_r++;

			if(len_r+es.size()<=es.capacity())
			{
				for(const Edge &c : r)
					if(curr.find(c)==curr.end())
						es.push_back(c);
			}
			else es = ANN::util::to<edgelist>(std::forward<R>(r));

			return *this;
		}
	};

	class vinfo_edges{
	public:
		using raw_t = edgelist;

		const edgelist& get_edges() const{
			return edges;
		}
		edgelist& get_edges(){
			return edges;
		}
		const raw_t& get_edges_raw() const&{
			return edges;
		}
		raw_t&& get_edges_raw() &&{
			return std::move(edges);
		}
		void set_edges_raw(const raw_t &other){
			edges = other;
		}
		void set_edges_raw(raw_t &&other){
			edges = std::move(other);
		}

		vinfo_edges() noexcept = default;
		vinfo_edges(const edgelist &edges) : edges(edges){
		}
		vinfo_edges(edgelist &&edges) : edges(std::move(edges)){
		}

	private:
		raw_t edges;
	};

	class vinfo_trivial_ext : public vinfo_edges{
		constexpr static const bool has_default =
			std::is_default_constructible_v<ext_t>;
	public:
		using raw_t = std::conditional_t<
			has_default, ext_t, std::optional<ext_t>
		>;

		const ext_t& get_ext() const{
			if constexpr(has_default)
				return ext;
			else
				return *ext;
		}
		// const ext_t& get_ext() const& = delete;
		// const ext_t& get_ext() && = delete;
		const raw_t& get_ext_raw() const&{
			return ext;
		}
		raw_t&& get_ext_raw() &&{
			return std::move(ext);
		}
		void set_ext_raw(const raw_t &other){
			ext = other;
		}
		void set_ext_raw(raw_t &&other){
			ext = std::move(other);
		}

		vinfo_trivial_ext() noexcept = default;
		vinfo_trivial_ext(ext_t ext) :
			ext(std::move(ext))
		{}
		vinfo_trivial_ext(edgelist edges) :
			vinfo_edges(std::move(edges)), ext()
		{}
		vinfo_trivial_ext(ext_t ext, edgelist edges) : 
			vinfo_edges(std::move(edges)), ext(std::move(ext))
		{}

	private:
		raw_t ext;
	};

	class vinfo_both_shared : public vinfo_edges{
	public:
		using raw_t = shared_wrapper<ext_t>;

		const ext_t& get_ext() const{
			return *ext;
		}
		const raw_t& get_ext_raw() const&{
			return ext;
		}
		raw_t&& get_ext_raw() &&{
			return std::move(ext);
		}
		void set_ext_raw(const raw_t &other){
			ext = other;
		}
		void set_ext_raw(raw_t &&other){
			ext = std::move(other);
		}

		vinfo_both_shared() noexcept = default;
		vinfo_both_shared(ext_t ext) :
			ext(std::move(ext))
		{}
		vinfo_both_shared(edgelist edges) :
			vinfo_edges(std::move(edges))
		{}
		vinfo_both_shared(ext_t ext, edgelist edges) : 
			vinfo_edges(std::move(edges)), ext(std::move(ext))
		{}

	private:
		raw_t ext;
	};

	using vinfo = std::conditional_t<
		std::is_trivially_copyable_v<ext_t> && sizeof(ext_t)<=64,
		vinfo_trivial_ext,
		vinfo_both_shared
	>;

	struct node_entry{
		using key_t = nid_t;
		using val_t = vinfo;

		static bool comp(key_t a, key_t b){
			return a < b;
		}
	};

	// it is safe to uncomment the following code only if 
	// pam_map and cpam::diff_encoded_map has been modified to
	// return a refernce of val_t rather than a value copy from .find()
	using nodemap = pam_map<node_entry>;
	// using nodemap = cpam::pam_map<node_entry>;
	using map_entry = typename nodemap::Entry;
	/*
#ifdef USE_PAM_UPPER
	using nodemap = pam_map<node_entry>;
#else
#ifdef USE_DIFF_ENCODING
	using nodemap = cpam::diff_encoded_map<node_entry, 64>;
#else
	using nodemap = cpam::pam_map<node_entry>;
#endif
#endif
	*/

	template<bool IsConst>
	struct ptr_base{
	public:
		using ptr_t = const ext_t*;
		using ref_t = const ext_t&;

		ref_t operator*() const{
			return content->get_ext();
		}
		ptr_t operator->() const{
			return &operator*();
		}

	protected:
		const vinfo *content;

		ptr_base(const vinfo &info) : content(&info){
		}

		friend class graph_cpam;
	};

	nodemap nodes;

public:
	struct node_ptr : ptr_base<false>{
		using ptr_base<false>::ptr_base;
	};

	struct node_cptr : ptr_base<true>{
		using ptr_base<true>::ptr_base;
		node_cptr(const node_ptr &other) :
			ptr_base<true>(other.content){
		}
		node_cptr(node_ptr &&other) :
			ptr_base<true>(std::move(other.content)){
		}
	};

private:
	template <typename Iter, class F>
	void insert_vertices_batch(Iter begin, Iter end, F &&comb){
		auto key_less = [&](const auto& l, const auto& r){
			return map_entry::comp(map_entry::get_key(l), map_entry::get_key(r));
		};

		auto range = parlay::make_slice(begin, end);
		parlay::sort_inplace(range, key_less);
		nodes = nodemap::multi_insert_sorted(std::move(nodes), range, comb);
	}
	template <typename Iter>
	void insert_vertices_batch(Iter begin, Iter end){
		auto comb_default = [](auto &&cur, auto &&inc) -> decltype(auto){
			(void)cur;
			return std::forward<decltype(inc)>(inc);
		};
		insert_vertices_batch(begin, end, comb_default);
	}

	void insert_vertex_inplace(nid_t id, const vinfo &e){
		using entry_t = typename map_entry::entry_t;
		nodes.insert(entry_t(id, e));
	}
	void insert_vertex_inplace(nid_t id, vinfo &&e){
		using entry_t = typename map_entry::entry_t;
		nodes.insert(entry_t(id, std::move(e)));
	}

	template<class T>
	node_ptr add_node_impl(nid_t u, T &&ext){
		insert_vertex_inplace(u, vinfo(std::move(ext)));
		return get_node(u);
	}

public:
	graph_cpam() = default;
	graph_cpam(const graph_cpam&) = default;
	graph_cpam& operator=(const graph_cpam&) = default;

	// add the missing noexcept specifier in the underlying structures
	graph_cpam(graph_cpam &&other) noexcept :
		nodes(std::move(other.nodes)){
	}
	graph_cpam& operator=(graph_cpam &&other) noexcept{
		nodes = std::move(other.nodes);
		return *this;
	}

	node_ptr get_node(nid_t u){
		return {*nodes.find(u)};
	}
	node_cptr get_node(nid_t u) const{
		return {*nodes.find(u)};
	}

	// TODO: simplify the code
	const_edge_agent get_edges(node_cptr p) const{
		return {p.content->get_edges()};
	}
	decltype(auto) get_edges(nid_t u) const{
		return get_edges(get_node(u));
	}
	edge_agent get_edges(node_ptr p){
		return {p.content->get_edges()};
	}
	decltype(auto) get_edges(nid_t u){
		return get_edges(get_node(u));
	}

	template<class Iter>
	void set_edges(Iter begin, Iter end){
		using entry_t = typename map_entry::entry_t;
		auto n = std::distance(begin, end);
		auto vs_delayed = ANN::util::delayed_seq(n, [&](size_t i){
			auto &&[u,ea] = *(begin+i);
			// return entry_t(u, vinfo(std::forward<decltype(es)>(es)));
			using ea_t = std::remove_cv_t<std::remove_reference_t<decltype(ea)>>;
			if constexpr(std::is_base_of_v<const_edge_agent,ea_t>)
				return entry_t(u, vinfo(
					std::forward<decltype(ea)>(ea).get_ref()
				));
			else // a range type
				return entry_t(u, vinfo(
					ANN::util::to<edgelist>(std::forward<decltype(ea)>(ea))
				));
		});
		using vs_t = typename cm::seq<entry_t>;
		vs_t vs(vs_delayed.begin(), vs_delayed.end());

		auto comb = [](auto &&cur, auto &&inc){
			vinfo updated;
			updated.set_ext_raw(cur.get_ext_raw());
			updated.set_edges_raw(inc.get_edges_raw());
			return updated;
		};
		insert_vertices_batch(vs.begin(), vs.end(), comb);
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

	node_ptr add_node(nid_t nid, const ext_t &ext){
		return add_node_impl(nid, ext);
	}
	node_ptr add_node(nid_t nid, ext_t &&ext){
		return add_node_impl(nid, std::move(ext));
	}
	template<class Seq>
	void add_nodes(Seq &&ns){
		using entry_t = typename map_entry::entry_t;
		auto vs_delayed = ANN::util::delayed_seq(ns.size(), [&](size_t i){
			auto &&[u,ext] = ns[i];
			return entry_t(u, vinfo(std::forward<decltype(ext)>(ext)));
		});
		using vs_t = typename cm::seq<entry_t>;
		vs_t vs(vs_delayed.begin(), vs_delayed.end());

		insert_vertices_batch(vs.begin(), vs.end());
	}

	bool empty() const{
		// here we fix the missing const-qualifier
		return const_cast<nodemap&>(nodes).is_empty();
	}
	size_t num_nodes() const{
		return const_cast<nodemap&>(nodes).size();
	}

	// TODO: eliminate redundant code by deducing 'this' in C++23
	template<class F>
	void iter_each(F &&f) const{
		auto g = [&](const auto &entry){
			f(node_cptr(map_entry::get_val(entry)));
		};
		nodemap::foreach_seq(nodes, g);
	}
	template<class F>
	void iter_each(F &&f){
		auto g = [&](const auto &entry){
			f(node_ptr(map_entry::get_val(entry)));
		};
		nodemap::foreach_seq(nodes, g);
	}
	template<class F>
	void for_each(F &&f) const{
		auto g = [&](const auto &entry, size_t/*pos*/){
			f(node_cptr(map_entry::get_val(entry)));
		};
		nodemap::foreach_index(nodes, g);
	}
	template<class F>
	void for_each(F &&f){
		auto g = [&](const auto &entry, size_t/*pos*/){
			f(node_ptr(map_entry::get_val(entry)));
		};
		nodemap::foreach_index(nodes, g);
	}
	template<class F>
	void for_each_raw(F &&f) const{
		auto g = [&](auto &&...args){
			f(std::forward<decltype(args)>(args)...);
		};
		nodemap::foreach_raw(nodes, g);
	}
};

#endif // _ANN_TEST_ASPEN_HPP
