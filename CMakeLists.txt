cmake_minimum_required(VERSION 3.22)
project(PixelAudit VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PIXELAUDIT_ENABLE_OPENMP "Enable OpenMP acceleration" ON)

find_package(OpenCV 4.0 REQUIRED COMPONENTS core imgproc imgcodecs)

if(PIXELAUDIT_ENABLE_OPENMP)
  find_package(OpenMP)
endif()

add_library(pixelaudit_core
  src/core/types.cpp
  src/core/pipeline.cpp
  src/audits/gradient_pca_test.cpp
  src/audits/frequency_spectrum_test.cpp
  src/audits/jpeg_forensics_test.cpp
  src/audits/optical_photometric_test.cpp
  src/audits/sensor_noise_prnu_test.cpp
  src/audits/vanishing_point_test.cpp
  src/audits/latent_module_stub.cpp
)

target_include_directories(pixelaudit_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(pixelaudit_core
  PUBLIC
    ${OpenCV_LIBS}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(pixelaudit_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(pixelaudit_core PUBLIC PIXELAUDIT_HAS_OPENMP=1)
endif()

add_executable(pixelaudit_cli src/cli/main.cpp)
target_link_libraries(pixelaudit_cli PRIVATE pixelaudit_core)

include(CTest)
if(BUILD_TESTING)
  add_executable(pixelaudit_smoke_test tests/smoke_test.cpp)
  target_link_libraries(pixelaudit_smoke_test PRIVATE pixelaudit_core)
  add_test(NAME pixelaudit_smoke_test COMMAND pixelaudit_smoke_test)

  add_executable(pixelaudit_frequency_audit_test tests/frequency_audit_test.cpp)
  target_link_libraries(pixelaudit_frequency_audit_test PRIVATE pixelaudit_core)
  add_test(NAME pixelaudit_frequency_audit_test COMMAND pixelaudit_frequency_audit_test)

  add_executable(pixelaudit_vanishing_audit_test tests/vanishing_audit_test.cpp)
  target_link_libraries(pixelaudit_vanishing_audit_test PRIVATE pixelaudit_core)
  add_test(NAME pixelaudit_vanishing_audit_test COMMAND pixelaudit_vanishing_audit_test)

  add_executable(pixelaudit_sensor_optical_audit_test tests/sensor_optical_audit_test.cpp)
  target_link_libraries(pixelaudit_sensor_optical_audit_test PRIVATE pixelaudit_core)
  add_test(NAME pixelaudit_sensor_optical_audit_test COMMAND pixelaudit_sensor_optical_audit_test)
endif()
