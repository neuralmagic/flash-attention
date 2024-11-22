// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
//

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                     \
  [&] {                                                                        \
    if (COND) {                                                                \
      constexpr static bool CONST_NAME = true;                                 \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static bool CONST_NAME = false;                                \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define CAUSAL_LOCAL_SWITCH(CAUSAL_COND, LOCAL_COND, CAUSAL_CONST_NAME, LOCAL_CONST_NAME, ...) \
  [&] {                                                                                        \
    if (CAUSAL_COND) {                                                                         \
      constexpr static bool CAUSAL_CONST_NAME = true;                                          \
      constexpr static bool LOCAL_CONST_NAME = false;                                          \
      return __VA_ARGS__();                                                                    \
    } else if (LOCAL_COND) {                                                                   \
      constexpr static bool CAUSAL_CONST_NAME = false;                                         \
      constexpr static bool LOCAL_CONST_NAME = true;                                           \
      return __VA_ARGS__();                                                                    \
    } else {                                                                                   \
      constexpr static bool CAUSAL_CONST_NAME = false;                                         \
      constexpr static bool LOCAL_CONST_NAME = false;                                          \
      return __VA_ARGS__();                                                                    \
    }                                                                                          \
  }()

#define PREC_SWITCH(PRECTYPE, ...)                                             \
  [&] {                                                                        \
    if (PRECTYPE == 1) {                                                       \
      using kPrecType = cutlass::half_t;                                       \
      constexpr static bool kSoftFp16 = false;                                 \
      constexpr static bool kHybrid = false;                                   \
      return __VA_ARGS__();                                                    \
    } else if (PRECTYPE == 2) {                                                \
      using kPrecType = cutlass::float_e4m3_t;                                 \
      constexpr static bool kSoftFp16 = false;                                 \
      constexpr static bool kHybrid = false;                                   \
      return __VA_ARGS__();                                                    \
    } else if (PRECTYPE == 3) {                                                \
      using kPrecType = cutlass::float_e4m3_t;                                 \
      constexpr static bool kSoftFp16 = false;                                 \
      constexpr static bool kHybrid = true;                                    \
      return __VA_ARGS__();                                                    \
    } else if (PRECTYPE == 4) {                                                \
      using kPrecType = cutlass::float_e4m3_t;                                 \
      constexpr static bool kSoftFp16 = true;                                  \
      constexpr static bool kHybrid = false;                                   \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define HEADDIM_SWITCH(HEADDIM, ...)                                           \
  [&] {                                                                        \
    if (HEADDIM == 64) {                                                       \
      constexpr static int kHeadSize = 64;                                     \
      return __VA_ARGS__();                                                    \
    } else if (HEADDIM == 96) {                                                \
      constexpr static int kHeadSize = 96;                                     \
      return __VA_ARGS__();                                                    \
    } else if (HEADDIM == 128) {                                               \
      constexpr static int kHeadSize = 128;                                    \
      return __VA_ARGS__();                                                    \
    } else if (HEADDIM == 96) {                                                \
      constexpr static int kHeadSize = 96;                                     \
      return __VA_ARGS__();                                                    \
    } else if (HEADDIM == 256) {                                               \
      constexpr static int kHeadSize = 256;                                    \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define SEQLEN_SWITCH(USE_VAR_SEQ_LEN, SEQ_LEN_OUT_OF_BOUND_CHECK, ...)        \
  [&] {                                                                        \
    if (!USE_VAR_SEQ_LEN) {                                                    \
      if (SEQ_LEN_OUT_OF_BOUND_CHECK) {                                        \
        using kSeqLenTraitsType = FixedSeqLenTraits<true>;                     \
        return __VA_ARGS__();                                                  \
      } else {                                                                 \
        using kSeqLenTraitsType = FixedSeqLenTraits<false>;                    \
        return __VA_ARGS__();                                                  \
      }                                                                        \
    } else {                                                                   \
      using kSeqLenTraitsType = VarSeqLenTraits;                               \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }() 