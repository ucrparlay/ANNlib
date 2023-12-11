#ifndef _ANN_CUSTOM_HPP
#define _ANN_CUSTOM_HPP

#include <type_traits>

namespace ANN{
namespace external{

auto def_custom_tag();

template<typename T, typename=void>
struct lookup_custom_tag_impl;

template<typename T>
struct lookup_custom_tag_impl<T,std::void_t<decltype(def_custom_tag(),bool{})>>{
	using type = decltype(def_custom_tag());
};

template<typename T=void>
using lookup_custom_tag = lookup_custom_tag_impl<T>;

template<class Tag>
class custom;

} // namespace external

using external::custom;
using external::lookup_custom_tag;

} // namespace ANN

#endif // _ANN_CUSTOM_HPP
