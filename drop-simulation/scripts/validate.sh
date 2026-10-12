#!/usr/bin/env bash
# Valida cada version GPU contra la CPU y deja el error por cuadro en validation/vN_errors.csv.
# Uso: scripts/validate.sh
set -euo pipefail
cd "$(dirname "$0")/.."

for v in v0 v1 v2 v3; do
    bin=build/drop_simulation_gpu_$v
    if [[ ! -x "$bin" ]]; then
        echo "Se omite $v: no existe $bin"
        continue
    fi
    echo "== $v"
    "$bin" --validate --csv "validation/${v}_errors.csv" | sed -n '/--- Validacion/,$p'
done
