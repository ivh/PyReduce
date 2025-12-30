#!/bin/bash
cd /Users/tom/PyReduce
export PATH="/Users/tom/.local/bin:$PATH"
exec /Users/tom/.local/bin/uv run pyesorex "$@"
