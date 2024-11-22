#pragma once

#include <cute/tensor.hpp>

namespace cute {

template <int Mode>
struct PagedMode
{
  static constexpr int mode = Mode;
  int32_t const* page_table;
  cutlass::FastDivmod subpages_per_page_divmod; 

  template <class Offset>
  CUTE_HOST_DEVICE constexpr
  auto
  apply(Offset const& offset) const
  {
    static_assert(is_tuple<Offset>::value, "Offset must be a tuple.");
    static_assert(rank_v<Offset> >= Mode, "Offset must have at least Mode dimensions.");
    int page_idx, subpage_idx;
    subpages_per_page_divmod(page_idx, subpage_idx, get<Mode>(offset));
    return replace<Mode>(offset, 
        page_table[page_idx] * subpages_per_page_divmod.divisor + subpage_idx);
  }

  template <class Offset>
  CUTE_HOST_DEVICE constexpr
  auto
  operator()(Offset const& offset) const
  {
    return apply(offset);
  }
};

template <int Mode>
struct OverrideMode
{
  int& override_value;
  static constexpr int mode = Mode;

  template <class Offset>
  CUTE_HOST_DEVICE
  auto
  apply(Offset const& offset) const
  {
    static_assert(is_tuple<Offset>::value, "Offset must be a tuple.");
    static_assert(rank_v<Offset> >= Mode, "Offset must have at least Mode dimensions.");
    return replace<Mode>(offset, override_value);
  }

  template <class Offset>
  CUTE_HOST_DEVICE
  auto
  operator()(Offset const& offset) const
  {
    return apply(offset);
  }
};

template <int N>
CUTE_HOST_DEVICE void print(PagedMode<N> const&)
{
  printf("PM<%d>", N);
}

template <class ShapeA, class StrideA,
          int Mode>
CUTE_HOST_DEVICE constexpr
auto
composition(Layout<ShapeA,StrideA> const& a,
            PagedMode<Mode>        const& b)
{
  return composition(b, Int<0>{}, a);
}

template <class ShapeA, class StrideA,
          int Mode>
CUTE_HOST_DEVICE constexpr
auto
composition(Layout<ShapeA,StrideA> const& a,
            OverrideMode<Mode>     const& b)
{
  return composition(b, Int<0>{}, a);
}


template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_mode_paged(Tensor<Engine, Layout> const& t, 
                int32_t const* page_table,
                cutlass::FastDivmod const& subpages_per_page_divmod)
{
  using Mode = decltype(stride<Is...>(t.layout()).mode());
  return t.compose(
    PagedMode<int(Mode{})>{page_table, subpages_per_page_divmod});
}


template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_mode_overridable(Tensor<Engine, Layout> const& t, int& override_value)
{
  using Mode = decltype(stride<Is...>(t.layout()).mode());
  return t.compose(OverrideMode<int(Mode{})>{override_value});
}


template <int Mode, class Engine, class O, class B, class... Coords>
CUTE_HOST_DEVICE constexpr
auto
get_original_mode_value(Tensor<Engine, ComposedLayout<OverrideMode<Mode>, O, B>> const& t, Coords const&... cs)
{
    auto l = t.layout();
    return get<Mode>((t.data() + l.layout_b()(cs...) + l.offset()).coord_);
}


} // namespace cute
