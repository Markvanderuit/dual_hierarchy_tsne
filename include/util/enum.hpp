#pragma once

#include <array>
#include <type_traits>

namespace tsne {
  namespace detail {
    // Cast enum value to underlying type of enum class (eg. int)
    template <typename ETy>
    constexpr inline
    typename std::underlying_type<ETy>::type underlying(ETy e) noexcept {
      return static_cast<typename std::underlying_type<ETy>::type>(e);
    }
  } // detail
    
  // Array class using Enum as indices
  // For a used enum E, E::Length must be a member of the enum
  template <typename ETy, typename Ty>
  class EnumArray : public std::array<Ty, detail::underlying(ETy::Length)> {
  public:
    constexpr inline
    const Ty& operator()(ETy e) const {
      return this->operator[](detail::underlying<ETy>(e));
    }

    constexpr inline
    Ty& operator()(ETy e) {
      return this->operator[](detail::underlying<ETy>(e));
    }
  };
} // tsne