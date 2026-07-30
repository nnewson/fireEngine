#!/usr/bin/env bash
# SH-03 slice 6 — the shadow-LOD calibration sweep, as a script rather than a remembered procedure.
#
# Two questions, measured separately:
#
#   BUDGET (quality/cost)  How much shadow does a budget cost, and what does it save? Measured on
#                          the STATIC scene, which is the only reproducible one — the animated scene
#                          advances on wall-clock time, so frame N differs between runs.
#   RATIO  (stability)     Does the hysteresis dead band remove chatter? Measured on the ANIMATED
#                          scene, aggregated over a whole run, because a still frame cannot tell
#                          "settled" from "caught between flickers".
#
# The reference is `--no-shadow-lod`, NOT `--no-lod` and NOT a tiny budget:
#   * `--no-lod` also disables FORWARD LOD, so the images differ in visible geometry and the
#     measurement stops being about shadows (this mistake was made, and the first table discarded).
#   * a tiny budget still RUNS selection, and any cut whose estimated deviation is 0 remains
#     eligible — "almost always LOD0" is not "LOD0".
#
# The quality metric is deliberately NOT whole-image PSNR. The forward image is identical between
# runs, so every difference is shadow error, but shadow error is localised to silhouette edges and
# an image-wide average dilutes it into invisibility. Instead we compare the SHADOW VISIBILITY
# image (`--debug-shadow`) and count pixels whose shadow state actually differs — plus the
# worst-pixel delta, which is what a viewer notices at an edge.
#
# SCOPE: `--debug-shadow` visualises the PRIMARY DIRECTIONAL visibility, and the triangle figures
# below are read from the cascade row, so this calibrates the CSM. Spot, point and self views share
# the constant but are not measured here; their savings are visible in the panel's other rows, and
# their quality is not. Widening this to the punctual families needs a per-family visibility view.
#
# Usage:  cd build && ../tools/shadow_lod_sweep.sh [output-dir]
set -euo pipefail
# Job control off: otherwise the shell prints its own "Terminated" notice when a backgrounded run is
# reaped, which reads like a failure in the middle of a results table.
set +m

OUT="${1:-/tmp/shadow-lod-sweep}"
SCENE="shadow_lod/ShadowLodDemo.gltf"
MOTION="shadow_lod/ShadowLodMotionDemo.gltf"
SKY="nightbox.hdr"
FRAME=16              # past IBL/swapchain settling; the runbook's reference frame
BUDGETS="0.5 1 2 4 8 16"
SELECTED_BUDGET=1     # kShadowLodPixelBudget: the dead band is measured where the engine runs
RATIOS="1.0 0.75 0.5"
SECONDS_PER_RATIO=12  # ~600-700 frames, enough for a rate rather than an anecdote
MIN_FRAMES=200        # below this a "rate" is noise; a vacuous zero must fail, not reassure
DIFF_THRESHOLD=8      # 8/255 luma: below this two shadow images agree to within dithering

command -v ffmpeg >/dev/null || { echo "ffmpeg is required for the image metrics" >&2; exit 1; }
[ -x ./fireEngineApp ] || { echo "run me from the build directory" >&2; exit 1; }
mkdir -p "$OUT"

# --- shadow mask: the shadow-visibility debug view, which isolates what we are measuring ---
capture_mask() { # <output> <extra flags...>
  local out="$1"; shift
  ./fireEngineApp "$SCENE" "$SKY" --no-taa --debug-shadow \
      --capture-frame "$FRAME" --capture "$out" "$@" >/dev/null 2>&1
}

# The reference's SHADOWED AREA, defined by what the shadows actually darken rather than by
# brightness. Counting "pixels darker than half" would fold in the night skybox and every dark
# material, inflating the denominator and flattering every percentage; here the area is the pixels
# that DIFFER between the full-detail reference and the same view with shadows switched off.
shadowed_fraction() { # <reference-visibility> <no-shadows-visibility>
  local ref="$1" unshadowed="$2"
  local avg
  avg=$(ffmpeg -hide_banner -i "$ref" -i "$unshadowed" \
    -lavfi "blend=all_mode=difference,format=gray,geq=lum='if(gt(lum(X,Y),$DIFF_THRESHOLD),255,0)',signalstats,metadata=print:key=lavfi.signalstats.YAVG" \
    -f null - 2>&1 | grep -o "YAVG=[0-9.]*" | head -1 | cut -d= -f2)
  awk -v a="$avg" 'BEGIN{ printf "%.8f", a/255.0 }'
}

# Fraction of pixels whose shadow state differs by more than DIFF_THRESHOLD, and the worst single-
# pixel delta.
#
# Reported TWICE, because the two denominators answer different questions: as a fraction of the
# frame (how much of the picture is wrong) and as a fraction of the shadowed area measured above
# (how much of the SHADOW is wrong). The second is the one that does not shrink just because a scene
# happens to be mostly empty floor.
mask_metrics() { # <reference> <candidate> <reference-shadowed-fraction>
  local ref="$1" cand="$2" shadowed="$3"
  local differing worst
  differing=$(ffmpeg -hide_banner -i "$ref" -i "$cand" \
    -lavfi "blend=all_mode=difference,format=gray,geq=lum='if(gt(lum(X,Y),$DIFF_THRESHOLD),255,0)',signalstats,metadata=print:key=lavfi.signalstats.YAVG" \
    -f null - 2>&1 | grep -o "YAVG=[0-9.]*" | head -1 | cut -d= -f2)
  worst=$(ffmpeg -hide_banner -i "$ref" -i "$cand" \
    -lavfi "blend=all_mode=difference,format=gray,signalstats,metadata=print:key=lavfi.signalstats.YMAX" \
    -f null - 2>&1 | grep -o "YMAX=[0-9.]*" | head -1 | cut -d= -f2)
  # YAVG of a 0/255 mask is 255 * fraction-of-pixels-differing.
  awk -v a="$differing" -v w="$worst" -v s="$shadowed" \
    'BEGIN{ f = a / 255.0; rel = 0; if (s > 0) rel = 100.0 * f / s;
            printf "%.4f %.3f %s", f * 100.0, rel, w }'
}

echo "=== budget sweep (ratio 1.0, static scene, reference = --no-shadow-lod) ==="
capture_mask "$OUT/ref-mask.png" --no-shadow-lod
capture_mask "$OUT/ref-mask-again.png" --no-shadow-lod          # the metric's own noise floor
capture_mask "$OUT/unshadowed-mask.png" --no-shadow-lod --no-shadows
./fireEngineApp "$SCENE" "$SKY" --no-taa --no-shadow-lod \
    --capture-frame "$FRAME" --capture "$OUT/ref-colour.png" >/dev/null 2>&1

REF_SHADOWED=$(shadowed_fraction "$OUT/ref-mask.png" "$OUT/unshadowed-mask.png")
# A zero shadowed area would make every relative error 0/0 -> 0.00% and the whole table would read
# as a pass. That happens whenever the reference and the no-shadows control come out identical: no
# sun, shadows already off, a scene with no casters, or a capture that failed.
if awk -v s="$REF_SHADOWED" 'BEGIN{ exit (s > 0) ? 0 : 1 }'; then :; else
  echo "FAIL: the reference and --no-shadows captures are identical, so the shadowed area is 0." >&2
  echo "      Every relative error would divide by zero and report 0.00% — check the sun, the" >&2
  echo "      casters, and $OUT/{ref,unshadowed}-mask.png before trusting any row." >&2
  exit 1
fi
awk -v s="$REF_SHADOWED" 'BEGIN{ printf "shadowed area: %.2f%% of the frame (measured, not assumed)\n", s*100 }'
# Two identical runs: anything at or below this is the measurement's own noise, not a difference
# the budget caused.
read -r npct nrel nworst <<<"$(mask_metrics "$OUT/ref-mask.png" "$OUT/ref-mask-again.png" "$REF_SHADOWED")"
printf "noise floor (reference vs itself): %s%% of frame, %s%% of shadow, worst %s\n\n" \
    "$npct" "$nrel" "$nworst"

printf "%-8s %-14s %-16s %-10s %s\n" budget "% of frame" "% of shadow" "worst px" "note"
for b in $BUDGETS; do
  capture_mask "$OUT/mask-$b.png" --shadow-budget "$b" --shadow-ratio 1.0
  ./fireEngineApp "$SCENE" "$SKY" --no-taa --shadow-budget "$b" --shadow-ratio 1.0 \
      --capture-frame "$FRAME" --capture "$OUT/colour-$b.png" >/dev/null 2>&1
  # An amplified difference image, so "thin edge slivers" vs "displaced silhouette" is a thing you
  # look at rather than a claim you make.
  ffmpeg -hide_banner -y -i "$OUT/ref-mask.png" -i "$OUT/mask-$b.png" \
    -lavfi "blend=all_mode=difference,format=gray,geq=lum='min(255,6*lum(X,Y))'" \
    "$OUT/diff-$b.png" >/dev/null 2>&1
  read -r pct rel worst <<<"$(mask_metrics "$OUT/ref-mask.png" "$OUT/mask-$b.png" "$REF_SHADOWED")"
  printf "%-8s %-14s %-16s %-10s %s\n" "$b" "$pct" "$rel" "$worst" "diff-$b.png"
done

echo
echo "=== triangles kept (read the cascade group row of the Shadows panel) ==="
echo "Panel captures below; the cascade row's 'Tris d/c' is drawn / full detail."
for b in $BUDGETS; do
  ./fireEngineApp "$SCENE" "$SKY" --overlay --no-taa --shadow-budget "$b" --shadow-ratio 1.0 \
      --capture-frame 24 --capture "$OUT/panel-$b.png" >/dev/null 2>&1
  echo "  $OUT/panel-$b.png"
done

echo
echo "=== dead-band sweep (animated scene, aggregated over a run) ==="
printf "%-8s %-8s %-14s %-14s %s\n" ratio frames transitions reversals "per 100 frames"
for r in $RATIOS; do
  log="$OUT/movement-$r.log"
  # At the SELECTED budget, not an arbitrary one: chatter depends on which thresholds casters sit
  # near, so a ratio measured at a different budget does not reproduce the committed decision.
  FE_LOG=render:debug ./fireEngineApp "$MOTION" "$SKY" --no-taa \
      --shadow-budget "$SELECTED_BUDGET" --shadow-ratio "$r" >"$log" 2>&1 &
  pid=$!
  sleep "$SECONDS_PER_RATIO"
  # THE LIVENESS CONTRACT, same as the repository's render smoke: the process must still be alive
  # when we signal it, and must exit 143 (SIGTERM) afterwards. Without this an early crash — or a
  # run that never rendered — yields an empty log, and the awk below happily prints 0.00 per 100
  # frames, which reads exactly like "no chatter".
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "FAIL ratio $r: the run exited before it was signalled (see $log)" >&2
    exit 1
  fi
  kill "$pid" 2>/dev/null || true
  # stderr is dropped only around the reap: bash prints its own "Terminated" notice there, which
  # lands in the middle of the results table and reads like a failure.
  rc=0; wait "$pid" 2>/dev/null || rc=$?
  if [ "$rc" -ne 143 ]; then
    echo "FAIL ratio $r: expected exit 143 from our SIGTERM, got $rc (see $log)" >&2
    exit 1
  fi
  frames=$(grep -c "shadow-lod movement" "$log" || true)
  if [ "$frames" -lt "$MIN_FRAMES" ]; then
    echo "FAIL ratio $r: only $frames movement records (need >= $MIN_FRAMES) — a zero rate here" \
         "would be vacuous, not evidence" >&2
    exit 1
  fi
  awk -v r="$r" '
    /shadow-lod movement/ {
      match($0, /transitions=[0-9]+/); t += substr($0, RSTART+12, RLENGTH-12)
      match($0, /reversed=[0-9]+/);    v += substr($0, RSTART+9,  RLENGTH-9)
      n++
    }
    END { printf "%-8s %-8d %-14d %-14d %.2f / %.2f\n", r, n, t, v, n?100*t/n:0, n?100*v/n:0 }
  ' "$log"
done
echo
echo "REVERSALS are the chatter signal: a caster returning to its previous level while that level"
echo "change is still RECENT (within the resolver's 30-commit window, ~0.5s at 60Hz) — i.e. sitting"
echo "on a threshold rather than travelling across it. A return after a long hold is ordinary motion"
echo "and is deliberately not counted, because a periodic animation walks levels back and forth."
echo "Plain transitions include that motion, which no dead band can or should remove, so a ratio is"
echo "only justified by the reversal column."
echo
echo "Artefacts in $OUT"
