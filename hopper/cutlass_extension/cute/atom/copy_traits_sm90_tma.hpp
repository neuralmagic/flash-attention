#pragma once

#include <cute/atom/copy_traits_sm90_tma.hpp>

namespace cute
{

// This is identical to `TMA_LOAD_Unpack` in `cute/atom/copy_traits_sm90_tma.hpp`
//  except for how the src_coord is determined. In this version we use `src(_0{})`
//  to get the src_coord for composed layouts since we can't assume slice_and_offset
//  positioned `src.data().coord_` and this allows us to deal with any 
//  non-linearities in the composed layout (since `src(_0{})` will force the 
//  non-linear layout code to be executed).
template <class CopyOp>
struct TMA_LOAD_Unpack_Composed
{
  template <class... Args,
            class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    auto src_coord = [&]() {
        if constexpr(is_composed_layout<SLayout>::value)
            return src(_0{});
        else 
            return src.data().coord_;
    }();

    if constexpr (detail::is_prefetch<CopyOp>) {
      return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                   traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                   src_coord, tuple_seq<decltype(src_coord)>{});
    } else {
      static_assert(is_smem<TD>::value, "SM90_TMA_LOAD requires the destination be shared memory.");
      void* dst_ptr = cute::raw_pointer_cast(dst.data());
#if 0
      if (thread0()) {
      auto [c0,c1,c2,c3,c4] = append<5>(src_coord, 0);
      printf("THR (%d,%d,%d) BLK (%d,%d,%d) TMACRD (%d,%d,%d,%d,%d) SMEMADDR (%p)\n",
            threadIdx.x, threadIdx.y, threadIdx.z,
            blockIdx.x, blockIdx.y, blockIdx.z,
            int32_t(c0), int32_t(c1), int32_t(c2), int32_t(c3), int32_t(c4), dst_ptr);
      }
#endif
      return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                   traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                   make_tuple(dst_ptr), seq<0>{},
                                   src_coord, tuple_seq<decltype(src_coord)>{});
    }
  }
};

//
// Tags for TMA copy ops that should dispatch to `TMA_LOAD_Unpack_Composed` 
//  instead of `TMA_LOAD_Unpack`
//

struct SM90_TMA_LOAD_COMPOSED_OP : SM90_TMA_LOAD_OP {};
struct SM90_TMA_LOAD_MULTICAST_COMPOSED_OP : SM90_TMA_LOAD_MULTICAST_OP {};

template <class NumBitsPerTMA>
struct Copy_Traits<SM90_TMA_LOAD_COMPOSED_OP, NumBitsPerTMA>
     : TMA_LOAD_Unpack_Composed<SM90_TMA_LOAD_COMPOSED_OP>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD arguments
  tuple<
  TmaDescriptor const*,
  uint64_t*, // smem mbarrier
  uint64_t   // cache hint
  > const opargs_;
};

template <class NumBitsPerTMA>
struct Copy_Traits<SM90_TMA_LOAD_MULTICAST_COMPOSED_OP, NumBitsPerTMA>
     : TMA_LOAD_Unpack_Composed<SM90_TMA_LOAD_MULTICAST_COMPOSED_OP>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD_MULTICAST arguments
  tuple<
  TmaDescriptor const*,
  uint64_t*, // smem mbarrier
  uint16_t,  // multicast mask
  uint64_t   // cache hint
  > const opargs_;
};


//
// The following utilties can be used to tag tma copy atoms where we want to 
//  use `TMA_LOAD_Unpack_Composed` instead of `TMA_LOAD_Unpack`. 
//  (currenlty the only use cases if for tma tensors composed with PagedMode)
//

template<class NumBitsPerTMA, class CopyInternalType>
CUTE_HOST_DEVICE constexpr
Copy_Atom<Copy_Traits<SM90_TMA_LOAD_COMPOSED_OP, NumBitsPerTMA>, CopyInternalType>
convert_tma_atom_to_support_compound_layouts(
    Copy_Atom<Copy_Traits<SM90_TMA_LOAD_OP, NumBitsPerTMA>, CopyInternalType> const& a) {
    return {{{}, a.opargs_}};
}

template<class NumBitsPerTMA, class CopyInternalType>
CUTE_HOST_DEVICE constexpr
Copy_Atom<Copy_Traits<SM90_TMA_LOAD_MULTICAST_COMPOSED_OP, NumBitsPerTMA>, CopyInternalType>
convert_tma_atom_to_support_compound_layouts(
    Copy_Atom<Copy_Traits<SM90_TMA_LOAD_MULTICAST_OP, NumBitsPerTMA>, CopyInternalType> const& a) {
    return {{{}, a.opargs_}};
}

} // namespace cute
