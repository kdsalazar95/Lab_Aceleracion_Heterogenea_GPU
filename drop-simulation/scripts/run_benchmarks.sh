#!/usr/bin/env bash
# Corre cada version N veces y guarda tiempo y pasos/s en results/benchmarks.csv.
# Uso: scripts/run_benchmarks.sh [N]   (N = 3 por defecto)
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-3}"
VERSIONS=(cpu v0 v1 v2 v3)
CSV=results/benchmarks.csv

mkdir -p results
echo "version,run,time_s,steps_per_s" > "$CSV"

for v in "${VERSIONS[@]}"; do
    if [[ "$v" == cpu ]]; then bin=build/drop_simulation; else bin=build/drop_simulation_gpu_$v; fi
    if [[ ! -x "$bin" ]]; then
        echo "Se omite $v: no existe $bin"
        continue
    fi
    for run in $(seq 1 "$N"); do
        out="$("$bin")"
        time_s="$(awk '/^Tiempo:/ {print $2}' <<< "$out")"
        steps="$(awk '/^Rendimiento:/ {print $2}' <<< "$out")"
        echo "$v,$run,$time_s,$steps" >> "$CSV"
        echo "$v corrida $run: $time_s s, $steps pasos/s"
    done
done

echo
echo "Promedios:"
awk -F, 'NR > 1 {t[$1] += $3; s[$1] += $4; n[$1]++; if (!($1 in o)) o[$1] = ++k}
         END {for (v in o) ord[o[v]] = v
              for (i = 1; i <= k; i++) {v = ord[i]; printf "%-4s %8.3f s %9.2f pasos/s\n", v, t[v]/n[v], s[v]/n[v]}}' "$CSV"
echo "Detalle: $CSV"
