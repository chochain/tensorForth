#!/bin/bash
# Run the example Forth scripts through ten4 on the simulated H100 and compare
# the printed results with the values given in the scripts and the README.
# usage: check.sh path/to/ten4
set -u
TEN4=$(realpath "$1")
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$(mktemp -d)
fail=0
n=0

# t4_20a without the 512x1024 benchmark and random-inverse sections
sed '12,41d' "$ROOT/examples/t4_20a.4th" > "$OUT/t4_20a_small.4th"

run() {  # name script
  timeout -s KILL 600 vgpu run --gpu nvidia/h100 "$TEN4" < "$2" > "$OUT/$1.raw" 2>&1
  local rc=$?
  tr -s '[:space:]' ' ' < "$OUT/$1.raw" > "$OUT/$1.txt"
  echo "== $1: rc=$rc, $(wc -l < "$OUT/$1.raw") lines"
  if [ $rc -ne 0 ]; then echo "FAIL $1: exit code $rc"; tail -20 "$OUT/$1.raw"; fail=1; fi
  # the run must really be on the simulated H100
  for s in "1 x nvidia/h100" "tensorForth v4.0 done."; do
    if ! grep -Fq -- "$s" "$OUT/$1.txt"; then echo "FAIL $1: missing '$s'"; fail=1; fi
  done
  if grep -Eq "CUDA error|cudaError|Segmentation|no CUDA-capable" "$OUT/$1.raw"; then
    echo "FAIL $1: CUDA error in output"; fail=1; fi
}
expect() {  # name text
  n=$((n+1))
  if grep -Fq -- "$2" "$OUT/$1.txt"; then echo "ok   $1: $2"
  else echo "FAIL $1: expected '$2'"; fail=1; fi
}

run t4_10a "$ROOT/examples/t4_10a.4th"
expect t4_10a "90 weather => too hot!"
expect t4_10a "70 weather => about right."
expect t4_10a "32 weather => too cold!"
expect t4_10a "54.9999 weather 55.0001 weather => too cold!, about right."
expect t4_10a "84.9999 weather 85.0001 weather => about right., too hot!"

run t4_20a "$OUT/t4_20a_small.4th"
expect t4_20a "matrix[2,2] = { { +6.0000 +6.0000 } { +15.0000 +15.0000 } }"
expect t4_20a "matrix[2,3] = { { +2.0000 +3.0000 +4.0000 } { +5.0000 +6.0000 +7.0000 } }"
expect t4_20a "matrix[2,3] = { { +0.0000 +1.0000 +2.0000 } { +3.0000 +4.0000 +5.0000 } }"
expect t4_20a "matrix[2,2] = { { +6.0000 +6.0000 } { +9.0000 +9.0000 } }"
expect t4_20a "matrix[2,2] = { { +3.0000 +3.0000 } { +4.5000 +4.5000 } }"

run t4_22a "$ROOT/examples/t4_22a.4th"
expect t4_22a "determinant = 6"
expect t4_22a "inverse = matrix[3,3] = { { +0.3333 +2.3333 -0.5000 } { -0.6667 -0.6667 +0.5000 } { +0.3333 -0.6667 -0.0000 } }"
expect t4_22a "luinv = matrix[3,3] = { { +0.3333 +2.3333 -0.5000 } { -0.6667 -0.6667 +0.5000 } { +0.3333 -0.6667 -0.0000 } }"
expect t4_22a "verify { 8 -1 -8 } => vector[3] = { +8.0000 -1.0000 -8.0000 }"
expect t4_22a "A@X = vector[3] = { +1.0000 +1.0000 +1.0000 }"

run t4_30a "$ROOT/examples/t4_30a.4th"
expect t4_30a "tensor[1,3,2,1] = { { { +0.1000 +0.2000 } { +0.3000 +0.4000 } { +0.5000 +0.6000 } } }"
expect t4_30a "vector[3] = { +1.0000 +2.0000 +3.0000 }"
expect t4_30a "tensor[1,1,3,1] = { { { +6.0000 +13.0000 +20.0000 } } }"

run t4_30b "$ROOT/examples/t4_30b.4th"
expect t4_30b "Model#loss: MSE= 0.658"
expect t4_30b "verify loss=0.658292=> 0.658292"

run t4_30c "$ROOT/examples/t4_30c.4th"
expect t4_30c "verify L0 W={ { +0.1359 +0.1718 } { +0.2339 +0.2679 } } => tensor[1,2,2,1] = { { { +0.1359 +0.1718 } { +0.2339 +0.2679 } } }"
expect t4_30c "verify L0 B={ +0.0680 +0.0287 } => vector[2] = { +0.0680 +0.0287 }"

echo "$n checks"
rm -rf "$OUT"
[ $fail -eq 0 ] || { echo "FAILED"; exit 1; }
echo "PASSED"
