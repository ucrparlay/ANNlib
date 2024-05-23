#ifndef _ANN_UTIL_SEQ_HPP
#define _ANN_UTIL_SEQ_HPP

#include <cstdint>
#include <limits>
#include <type_traits>
#include "iter.hpp"
#include "util.hpp"

namespace ANN::util{

template<typename F=identity>
class delayed_seq
{
private:
	using index_t = size_t;

	const index_t bound;
	F f;

public:
	class const_iterator : public enable_rand_iter<
		const_iterator, index_t, const F*>
	{
		using base = enable_rand_iter<
			const_iterator, index_t, const F*
		>;

	public:
		typedef std::invoke_result_t<F,index_t> value_type;
		typedef materialized_ptr<value_type> pointer;
		typedef value_type reference;

		const_iterator():
			base(std::numeric_limits<index_t>::max(), nullptr){
		}
		const_iterator(index_t index, const F *pf) :
			base(index, pf){
		}

		reference operator*() const{
			const F *pf = base::get_baseinfo();
			return (*pf)(base::get_pos());
		}
		pointer operator->() const{
			return {operator*()};
		}
	};

	using size_type = index_t;
	using difference_type = typename const_iterator::difference_type;
	using value_type = typename const_iterator::value_type;
	using pointer = typename const_iterator::pointer;
	using reference = typename const_iterator::reference;
	using const_pointer = pointer;
	using const_reference = reference;

	delayed_seq(index_t end, F f={})
		: bound(end), f(std::move(f)){
	}
	const_iterator begin() const{
		return const_iterator(0, &f);
	}
	const_iterator end() const{
		return const_iterator(bound, &f);
	}
	size_type size() const{
		return bound;
	}
	const_reference operator[](index_t index) const{
		return begin()[index];
	}
};

} // namespace ANN::util

#endif // _ANN_UTIL_SEQ_HPP
