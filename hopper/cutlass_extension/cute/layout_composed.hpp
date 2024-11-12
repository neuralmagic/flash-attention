#pragma once

#include <cute/layout_composed.hpp>

namespace cute {

template <int... Is, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
select(ComposedLayout<A,O,B> const& c)
{
  return composition(c.layout_a(), c.offset(), select<Is...>(c.layout_b()));
}

} // namespace cute
