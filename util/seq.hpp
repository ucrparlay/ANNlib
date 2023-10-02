#ifndef _ANN_UTIL_SEQ_HPP
#define _ANN_UTIL_SEQ_HPP

#include <cstdint>
#include <iterator>
#include "util.hpp"

namespace ANN::util{

namespace detail{

template<typename T>
class pointer_wrapper{
	using value_type = std::remove_reference_t<T>;
public:
	T value;

	value_type* operator->(){
		return &value;
	}
	value_type& operator*(){
		return value;
	}
};

template<typename T>
class iterator_base{
public:
	typedef std::random_access_iterator_tag iterator_category;
	typedef ptrdiff_t difference_type;
	typedef T value_type;
	typedef pointer_wrapper<value_type> pointer;
	typedef value_type reference;

	iterator_base(const T &data) : position(data){}
	iterator_base(const iterator_base &other) : position(other.position){}

	bool operator==(const iterator_base &rhs) const{
		return position==rhs.position;
	}
	bool operator!=(const iterator_base &rhs) const{
		return position!=rhs.position;
	}

	reference operator*() const{
		return position;
	}
	pointer operator->() const{
		return {operator*()};
	}

	iterator_base& operator+=(difference_type offset){
		return position+=offset, *this;
	}
	iterator_base& operator++(){ // ++a
		return ++position, *this;
	}
	iterator_base operator++(int){ // a++
		return position++;
	}
	iterator_base operator+(difference_type offset) const{
		return position+offset;
	}

	iterator_base& operator-=(difference_type offset){
		return position-=offset, *this;
	}
	iterator_base& operator--(){
		return ++position, *this;
	}
	iterator_base operator--(int){
		return position++;
	}
	iterator_base operator-(difference_type offset) const{
		return position-offset;
	}
	difference_type operator-(const iterator_base &rhs) const{
		return position-rhs.position;
	}

	reference operator[](difference_type offset) const{
		return *operator+(offset);
	}

	bool operator<(const iterator_base &rhs) const{
		return position<rhs.position;
	}
	bool operator>(const iterator_base &rhs) const{
		return position>rhs.position;
	}
	bool operator<=(const iterator_base &rhs) const{
		return position<=rhs.position;
	}
	bool operator>=(const iterator_base &rhs) const{
		return position>=rhs.position;
	}

protected:
	T position;
};

} // namespace detail

template<typename F=identity>
class delayed_seq
{
private:
	using index_t = size_t;

	const index_t bound;
	F f;

public:
	class const_iterator : public detail::iterator_base<index_t>{
		using base = detail::iterator_base<index_t>;

		const F *pf;

		const_iterator(base &&b, const F *pf) :
			base(b), pf(pf){
		}

	public:
		typedef base::difference_type difference_type;
		typedef std::invoke_result_t<F,index_t> value_type;
		typedef detail::pointer_wrapper<value_type> pointer;
		typedef value_type reference;

		const_iterator():
			base(std::numeric_limits<index_t>::max()), pf(nullptr){
		}
		const_iterator(index_t index, const F *pf) :
			base(index), pf(pf){
		}

		bool operator==(const const_iterator &rhs) const{
			return base::operator==(rhs) && pf==rhs.pf;
		}
		bool operator!=(const const_iterator &rhs) const{
			return base::operator!=(rhs) || pf!=rhs.pf;
		}

		reference operator*() const{
			return (*pf)(base::operator*());
		}
		pointer operator->() const{
			return {operator*()};
		}

		const_iterator& operator+=(difference_type offset){
			return static_cast<const_iterator&>(base::operator+=(offset));
		}
		const_iterator& operator++(){ // ++a
			return static_cast<const_iterator&>(base::operator++());
		}
		const_iterator operator++(int _){ // a++
			return {base::operator++(_), pf};
		}
		const_iterator operator+(difference_type offset) const{
			return {base::operator+(offset), pf};
		}

		reference operator[](difference_type offset) const{
			return *operator+(offset);
		}

		const_iterator& operator-=(difference_type offset){
			return static_cast<const_iterator&>(base::operator-=(offset));
		}
		const_iterator& operator--(){
			return static_cast<const_iterator&>(base::operator--());
		}
		const_iterator operator--(int _){
			return {base::operator--(_), pf};
		}
		const_iterator operator-(difference_type offset) const{
			return {base::operator-(offset), pf};
		}
		difference_type operator-(const const_iterator &rhs) const{
			return base::operator-(rhs);
		} // TODO: fix
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
	size_t size() const{
		return bound;
	}
	const_reference operator[](index_t index) const{
		return begin()[index];
	}
};

} // namespace ANN::util

#endif // _ANN_UTIL_SEQ_HPP
