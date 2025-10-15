cmake_minimum_required(VERSION 3.22.1)
project(QwenInstruct LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

set(JNI_LIBS_DIR "${CMAKE_CURRENT_LIST_DIR}/../jniLibs/${ANDROID_ABI}")
set(LLAMA_VENDOR_DIR "${CMAKE_CURRENT_LIST_DIR}/llama")
set(LLAMA_INCLUDE_DIR "${LLAMA_VENDOR_DIR}/include")
set(GGML_INCLUDE_DIR "${LLAMA_VENDOR_DIR}/ggml")
set(COMMON_INCLUDE_DIR "${LLAMA_VENDOR_DIR}/common")

if (NOT EXISTS "${JNI_LIBS_DIR}/libllama.so")
    message(FATAL_ERROR "Expected libllama.so in ${JNI_LIBS_DIR}; run scripts/build_llama_snapdragon830.sh (or the matching Snapdragon build script)")
endif()

if (NOT EXISTS "${JNI_LIBS_DIR}/libggml.so")
    message(FATAL_ERROR "Expected libggml.so in ${JNI_LIBS_DIR}; run scripts/build_llama_snapdragon830.sh (or the matching Snapdragon build script)")
endif()

include_directories(
        "${LLAMA_INCLUDE_DIR}"
        "${GGML_INCLUDE_DIR}"
        "${COMMON_INCLUDE_DIR}"
)

add_library(ggml SHARED IMPORTED)
set_target_properties(ggml PROPERTIES IMPORTED_LOCATION "${JNI_LIBS_DIR}/libggml.so")

add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES IMPORTED_LOCATION "${JNI_LIBS_DIR}/libllama.so")

add_library(native-lib SHARED qwen_bridge.cpp)

find_library(log-lib log)
find_library(android-lib android)
find_library(z-lib z)

if (log-lib STREQUAL "log-lib-NOTFOUND")
    message(FATAL_ERROR "Android log library not found")
endif()

if (android-lib STREQUAL "android-lib-NOTFOUND")
    message(FATAL_ERROR "Android native android lib not found")
endif()

if (z-lib STREQUAL "z-lib-NOTFOUND")
    message(FATAL_ERROR "zlib dependency not found")
endif()

set(LINK_LIBS
        llama
        ggml
        ${log-lib}
        ${android-lib}
        ${z-lib})

set_source_files_properties(qwen_bridge.cpp PROPERTIES COMPILE_FLAGS "-Wno-unused-parameter")

target_link_libraries(native-lib PRIVATE ${LINK_LIBS})

