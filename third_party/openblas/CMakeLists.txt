cmake_minimum_required(VERSION 3.18)

if (MSVC)
  add_library(openblas SHARED IMPORTED GLOBAL)
  set_target_properties(openblas PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/Win/bin/libopenblas.dll
    IMPORTED_IMPLIB  ${CMAKE_CURRENT_SOURCE_DIR}/Win/lib/libopenblas.dll.a
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/Win/Include
  )
else()
  find_package(OpenBLAS REQUIRED)
endif()