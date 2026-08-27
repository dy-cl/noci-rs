#!/usr/bin/env bash

# Test-only: pin matrixmultiply to SSE2 so ndarray .dot() is reproducible
# across CPU feature sets. Production builds retain native runtime dispatch.
export MMTEST_FEATURE=sse2

if [ "$#" -eq 0 ]; then
    exec cargo test --release
fi

exec "$@"
