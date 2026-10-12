#!/usr/bin/env bash
# Arma los videos del entregable a partir de los que genera cada version:
#   output/drop_simulation_final.mp4        version final (v1 por defecto), en H.264
#   output/drop_simulation_comparativo.mp4  CPU, v0, v1, v2 y v3 lado a lado
# Uso: scripts/make_final_video.sh            # final = v1
#      FINAL_VERSION=v3 scripts/make_final_video.sh
set -euo pipefail
cd "$(dirname "$0")/.."

FINAL_VERSION="${FINAL_VERSION:-v1}"
inputs=(output/drop_simulation.mp4)
for v in v0 v1 v2 v3; do inputs+=("output/drop_simulation_gpu_$v.mp4"); done

for f in "${inputs[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "Falta $f: corre primero scripts/run_benchmarks.sh" >&2
        exit 1
    fi
done

ffmpeg -y -loglevel error -i "output/drop_simulation_gpu_$FINAL_VERSION.mp4" \
    -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
    output/drop_simulation_final.mp4

# Cada video se reduce a 384x384; el texto de cada cuadro ya trae el nombre de la version.
ffmpeg -y -loglevel error \
    -i "${inputs[0]}" -i "${inputs[1]}" -i "${inputs[2]}" -i "${inputs[3]}" -i "${inputs[4]}" \
    -filter_complex "[0]scale=384:384[a];[1]scale=384:384[b];[2]scale=384:384[c];[3]scale=384:384[d];[4]scale=384:384[e];[a][b][c][d][e]hstack=inputs=5" \
    -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart \
    output/drop_simulation_comparativo.mp4

echo "Video final ($FINAL_VERSION): output/drop_simulation_final.mp4"
echo "Comparativo:               output/drop_simulation_comparativo.mp4"
