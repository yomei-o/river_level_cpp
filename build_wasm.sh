#!/bin/sh
# WASM デモ（docs/）をビルドする。EMSDK は環境変数で上書き可
EMSDK=${EMSDK:-/c/prog/emsdk/emsdk}
EMCC="$EMSDK/upstream/emscripten/emcc.exe"
[ -x "$EMCC" ] || EMCC="$EMSDK/upstream/emscripten/emcc"
cd "$(dirname "$0")"
"$EMCC" -O3 -std=c++17 wasm_api.cpp -o docs/river_level.js \
  -sMODULARIZE -sEXPORT_NAME=RiverLevel -sENVIRONMENT=web -sALLOW_MEMORY_GROWTH \
  -sEXPORTED_FUNCTIONS=_rl_frame_buffer,_rl_set_params,_rl_reset,_rl_results,_rl_process \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF64
