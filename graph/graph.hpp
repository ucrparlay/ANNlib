#ifndef _ANN_GRAPH_HPP
#define _ANN_GRAPH_HPP

#include <cstdint>
#include <type_traits>
#include "ANN.hpp"

namespace ANN{
namespace graph{

class base
{
	struct node;
public:
	using nid_t = uint32_t;
	struct node_ptr;
	struct node_cptr;
	// TODO: update interfaces
	node_ptr get_node(nid_t);
	node_cptr get_node(nid_t) const;
	template<class Seq>
	Seq get_edges(node_cptr) const;
	template<class Seq>
	Seq get_edges(nid_t) const;
	template<class Seq>
	void set_edges(nid_t, Seq&&);
	void add_node(nid_t);
	template<class Seq>
	void add_nodes(Seq&&);
	size_t num_nodes() const;
	void remove_node(nid_t);
	template<typename Iter>
	void remove_nodes(Iter, Iter);
};

template<class>
class shim;

enum class file_format{
	BUILTIN
};

namespace detail{

template<class T>
const static bool constexpr is_graph = std::is_base_of_v<base,T>;
/*
template<class T>
concept Graph = is_graph<T>
*/

} // namespace detail

template<class G>
void save(const G &g, const std::string &name, file_format type)
{
	static_assert(detail::is_graph<G>);
}

template<class G>
G load(const std::string &name, file_format type)
{
	static_assert(detail::is_graph<G>);
}

template<class To, class From>
To cast(From &&g)
{
	static_assert(detail::is_graph<From>&&detail::is_graph<To>);
}

} // namespace graph
} // namespace ANN

#endif // _ANN_GRAPH_HPP
