#pragma once

#include <cute/tensor_impl.hpp>

#include "cutlass_extension/cute/layout_composed.hpp"

namespace cute
{

template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
select(Tensor<Engine, Layout> const& t)
{
  return make_tensor(t.data(), select<Is...>(t.layout()));
}

} // namespace cute