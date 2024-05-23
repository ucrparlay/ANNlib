#ifndef _ANN_MAP_DIRECT_HPP
#define _ANN_MAP_DIRECT_HPP

#include <iterator>
#include <utility>
#include <type_traits>
#include <optional>
#include <ranges>
#include <unordered_map>
#include "map.hpp"
#include "util/seq.hpp"

namespace ANN::map{

template<typename Pid, typename Nid>
class direct{
	std::unordered_map<Nid,Pid> mapping;
public:
	template<typename Iter>
	void insert(Iter begin, Iter end){
		// use mapping.insert_range() in C++23
		const auto n = std::distance(begin, end);
		auto ps = util::delayed_seq(n, [&](size_t i){
			auto &&pid = *(begin+i);
			return std::pair<Nid,Pid>(
				Nid(pid), std::forward<decltype(pid)>(pid)
			);
		});
		mapping.insert(ps.begin(), ps.end());
	}
	template<class Ctr>
	void insert(Ctr &&c){
		if constexpr(std::is_rvalue_reference_v<Ctr&&>)
		{
			insert(
				std::make_move_iterator(c.begin()),
				std::make_move_iterator(c.end())
			);
		}
		else insert(c.begin(), c.end());
	}

	Nid insert(const Pid &pid){
		return mapping.insert({Nid(pid),pid}).first->first;
	}
	Nid insert(Pid &&pid){
		return mapping.insert({Nid(pid),std::move(pid)}).first->first;
	}

	template<typename Iter>
	void erase(Iter begin, Iter end){
		for(auto it=begin; it!=end; ++it)
			mapping.erase(Nid(*it));
	}

	void erase(Nid nid){
		mapping.erase(nid);
	}

	Pid get_pid(Nid nid) const{
		return mapping.find(nid)->second;
	}
	Nid get_nid(const Pid &pid) const{
		static_assert(std::is_convertible_v<Pid,Nid>);
		return Nid(pid);
	}

	Nid front_nid() const{
		return mapping.begin()->first;
	};

	// TODO: consider to remove it
	std::optional<Nid> find_nid(const Pid &pid) const{
		auto it = mapping.find(Nid(pid));
		if(it==mapping.end())
			return std::nullopt;
		return {it->first};
	}

	// TODO: use a more unambiguous name
	bool contain_nid(Nid nid) const{
		return mapping.contains(nid);
	}
};

} // namespace ANN::map

#endif // _ANN_MAP_DIRECT_HPP
