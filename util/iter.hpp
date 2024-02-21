#ifndef _ANN_UTIL_ITER_HPP
#define _ANN_UTIL_ITER_HPP

#include <cstdint>
#include <utility>
#include <iterator>
#include "util.hpp"

namespace ANN::util{
namespace detail{

template<typename P>
class rand_iter_base{
private:
	P pos;

protected:
	P& get_pos(){
		return pos;
	}
	const P& get_pos() const{
		return pos;
	}

protected:
	typedef std::random_access_iterator_tag iterator_category;
	typedef ptrdiff_t difference_type;

	rand_iter_base(const P &pos) : pos(pos){}

	bool operator==(const rand_iter_base &rhs) const{
		return pos==rhs.pos;
	}
	bool operator!=(const rand_iter_base &rhs) const{
		return pos!=rhs.pos;
	}

	rand_iter_base& operator+=(difference_type offset){
		return pos+=offset, *this;
	}
	rand_iter_base& operator++(){ // ++a
		return ++pos, *this;
	}
	rand_iter_base operator++(int){ // a++
		return pos++;
	}
	rand_iter_base operator+(difference_type offset) const{
		return pos+offset;
	}

	rand_iter_base& operator-=(difference_type offset){
		return pos-=offset, *this;
	}
	rand_iter_base& operator--(){
		return ++pos, *this;
	}
	rand_iter_base operator--(int){
		return pos++;
	}
	rand_iter_base operator-(difference_type offset) const{
		return pos-offset;
	}
	difference_type operator-(const rand_iter_base &rhs) const{
		return pos-rhs.pos;
	}

	bool operator<(const rand_iter_base &rhs) const{
		return pos<rhs.pos;
	}
	bool operator>(const rand_iter_base &rhs) const{
		return pos>rhs.pos;
	}
	bool operator<=(const rand_iter_base &rhs) const{
		return pos<=rhs.pos;
	}
	bool operator>=(const rand_iter_base &rhs) const{
		return pos>=rhs.pos;
	}
};

} // namespace detail

template<class D, typename P, typename B>
class enable_rand_iter :
	public detail::rand_iter_base<P>
{
private:
	using base = detail::rand_iter_base<P>;
	// TODO: use a less confusing name with baseinfo?
	B baseinfo;

protected:
	B& get_baseinfo(){
		return baseinfo;
	}
	const B& get_baseinfo() const{
		return baseinfo;
	}

	enable_rand_iter(base &&b, const B &baseinfo) :
		base(std::move(b)), baseinfo(baseinfo){
	}
	enable_rand_iter(const P &pos, const B &baseinfo) :
		base(pos), baseinfo(baseinfo){
	}
	enable_rand_iter(const P &pos, B &&baseinfo) :
		base(pos), baseinfo(std::move(baseinfo)){
	}

public:
	using typename base::iterator_category;
	using typename base::difference_type;
	// --- the types to define ---
	typedef void value_type;
	typedef void pointer;
	typedef void reference;

	bool operator==(const enable_rand_iter &rhs) const{
		return base::operator==(rhs) && baseinfo==rhs.baseinfo;
	}
	bool operator!=(const enable_rand_iter &rhs) const{
		return base::operator!=(rhs) || baseinfo!=rhs.baseinfo;
	}

	D& operator+=(difference_type offset){
		base::operator+=(offset);
		return static_cast<D&>(*this);
	}
	D& operator++(){ // ++iter
		static_assert(
			std::is_convertible_v<D*,enable_rand_iter*> &&
			sizeof(enable_rand_iter)==sizeof(D)
		);
		base::operator++();
		return static_cast<D&>(*this);
	}
	D operator++(int _){ // iter++
		static_assert(std::is_constructible_v<D,enable_rand_iter&&>);
		return enable_rand_iter(base::operator++(_), baseinfo);
	}
	D operator+(difference_type offset) const{
		return enable_rand_iter(base::operator+(offset), baseinfo);
	}

	D& operator-=(difference_type offset){
		base::operator-=(offset);
		return static_cast<D&>(*this);
	}
	D& operator--(){
		base::operator--();
		return static_cast<D&>(*this);
	}
	D operator--(int _){
		return enable_rand_iter(base::operator--(_), baseinfo);
	}
	D operator-(difference_type offset) const{
		return enable_rand_iter(base::operator-(offset), baseinfo);
	}

	difference_type operator-(const enable_rand_iter &rhs) const{
		return base::operator-(rhs);
	} // TODO: fix

	decltype(auto) operator[](difference_type offset) const{
		return *operator+(offset);
	}

	using base::operator<;
	using base::operator>;
	using base::operator<=;
	using base::operator>=;

	// --- the member functions to define ---
	reference operator*() const;
	pointer operator->() const;
};

template<class D, typename P>
class enable_rand_iter<D,P,empty> :
	public detail::rand_iter_base<P>
{
private:
	using base = detail::rand_iter_base<P>;

protected:
	empty get_baseinfo() const{
		return {};
	}

	enable_rand_iter(base &&b) :
		base(std::move(b)){
	}
	enable_rand_iter(const P &pos, empty={}) :
		base(pos){
	}

public:
	using typename base::iterator_category;
	using typename base::difference_type;
	// --- the types to define ---
	typedef void value_type;
	typedef void pointer;
	typedef void reference;

	using base::operator==;
	using base::operator!=;

	D& operator+=(difference_type offset){
		return static_cast<D&>(base::operator+=(offset));
	}
	D& operator++(){ // ++iter
		static_assert(
			std::is_convertible_v<D*,enable_rand_iter*> &&
			sizeof(enable_rand_iter)==sizeof(D)
		);
		return static_cast<D&>(base::operator++());
	}
	D operator++(int _){ // iter++
		static_assert(std::is_constructible_v<D,enable_rand_iter&&>);
		return enable_rand_iter(base::operator++(_));
	}
	D operator+(difference_type offset) const{
		return enable_rand_iter(base::operator+(offset));
	}

	D& operator-=(difference_type offset){
		return static_cast<D&>(base::operator-=(offset));
	}
	D& operator--(){
		return static_cast<D&>(base::operator--());
	}
	D operator--(int _){
		return enable_rand_iter(base::operator--(_));
	}
	D operator-(difference_type offset) const{
		return enable_rand_iter(base::operator-(offset));
	}

	difference_type operator-(const enable_rand_iter &rhs) const{
		return base::operator-(rhs);
	} // TODO: fix

	decltype(auto) operator[](difference_type offset) const{
		return *operator+(offset);
	}

	using base::operator<;
	using base::operator>;
	using base::operator<=;
	using base::operator>=;

	// --- the member functions to define ---
	reference operator*() const;
	pointer operator->() const;
};
// TODO: consider using [[no_unique_address]] for B to replace this specialization

} // namespace ANN::util

#endif // _ANN_UTIL_ITER_HPP
