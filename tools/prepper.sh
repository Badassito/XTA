#!/bin/bash
#SBATCH --job-name=XTA_Prep
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --output=%j.txt
#SBATCH --error=%j.txt

# Keep this script, prepare_pta_input.py and xta_edited_shape.py together.
# Submit from their directory, or set PREPPER_HOME to that directory.
# No GPU or training job is launched.
#
# New layout beneath TRAINING_ROOT:
#   images/DATASET/VIDEO_STEM.mkv
#   filtered/GENERATION/DATASET/{edits.json,source_manifest.json,*.npz}
#   labels/GENERATION/DATASET/{DATASET.nrrd,preparation.json}
#
# Examples:
#   sbatch prepper.sh
#   sbatch prepper.sh --validate-only
#   sbatch prepper.sh --smoke --smoke-script /path/to/pta_launcher.py
#   GENERATION=v1.2 sbatch prepper.sh
#   DATASET=OtherDataset VIDEO_STEM=OtherDataset_S1S0 sbatch prepper.sh
#
# Defaults reconstruct the edited Global at the full source dimensions.
# Original prediction NRRDs are not restored; their dimensions are recorded in
# the edit bundle and checked against the source video before preparation.
# Images stay in images/; labels contain the NRRD and provenance receipt only.
# With --smoke, copy the video and prepared mask to /tmp/pta-smoke and use that
# directory as PTA input. The copies remain there; the folder has no job ID.
#
# Environment overrides: ROOT, TRAINING_ROOT, GENERATION, DATASET, VIDEO_STEM,
# SOURCE_STEM, BUNDLE_DIR, SOURCE_VIDEO, OUTPUT_DIR, OUTPUT_STEM, MODE,
# PREPPER_HOME, VENV, PYTHON, LOG_ROOT, FFPROBE, BLOCK_MIB.
# Additional helper flags may be passed on the command line; later flags win.
# Select the volume through the environment variables, not positional arguments.
set -euo pipefail
umask 077

ROOT="${ROOT:-/project/community/sanghyus}"
TRAINING_ROOT="${TRAINING_ROOT:-${ROOT}/XTA_TRAINING}"
GENERATION="${GENERATION:-v1.1}"
DATASET="${DATASET:-F3_20_2_3_2026}"
VIDEO_STEM="${VIDEO_STEM:-${DATASET}_S1S0}"
SOURCE_STEM="${SOURCE_STEM:-${VIDEO_STEM}_8bit_Y}"
BUNDLE_DIR="${BUNDLE_DIR:-${TRAINING_ROOT}/filtered/${GENERATION}/${DATASET}}"
SOURCE_VIDEO="${SOURCE_VIDEO:-${TRAINING_ROOT}/images/${DATASET}/${VIDEO_STEM}.mkv}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAINING_ROOT}/labels/${GENERATION}/${DATASET}}"
OUTPUT_STEM="${OUTPUT_STEM:-${DATASET}}"
MODE="${MODE:-edited_shape}"
BLOCK_MIB="${BLOCK_MIB:-1024}"
PREPPER_HOME="${PREPPER_HOME:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
HELPER="${PREPPER_HOME}/prepare_pta_input.py"
VENV="${VENV:-${HOME}/venvs/ultralytics}"
if [[ "${VENV}" != "none" ]]; then
    [[ -f "${VENV}/bin/activate" ]] || { echo "Missing venv: ${VENV}; set VENV=none to use current Python." >&2; exit 1; }
    source "${VENV}/bin/activate"
fi
PYTHON="${PYTHON:-python}"
FFPROBE="${FFPROBE:-ffprobe}"
[[ -f "${HELPER}" ]] || { echo "Missing ${HELPER}; submit from its folder or set PREPPER_HOME." >&2; exit 1; }
command -v "${PYTHON}" >/dev/null || { echo "Python missing: ${PYTHON}" >&2; exit 1; }
if [[ -d "${HOME}/local/bin/ffmpeg-7.0.2-amd64-static" ]]; then
    export PATH="${HOME}/local/bin/ffmpeg-7.0.2-amd64-static:${PATH}"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG_ROOT="${LOG_ROOT:-${TRAINING_ROOT}/_preparation_logs}"
mkdir -p -- "${LOG_ROOT}"
RUN_DIR="$(mktemp -d "${LOG_ROOT}/prep-${SLURM_JOB_ID:-manual}-XXXXXXXX")"
exec > >(tee -a "${RUN_DIR}/launcher.log") 2>&1
echo "Prep logs: ${RUN_DIR}"
echo "Defaults: mode=${MODE}, edited Global, generation=${GENERATION}; CLI flags may override."
echo "Filtered bundle: ${BUNDLE_DIR}"
echo "Source video: ${SOURCE_VIDEO}"
echo "Label output: ${OUTPUT_DIR}/${OUTPUT_STEM}.nrrd"
"${PYTHON}" "${HELPER}" "${SOURCE_STEM}" \
    --root "${ROOT}" --training-root "${TRAINING_ROOT}" --generation "${GENERATION}" \
    --bundle-dir "${BUNDLE_DIR}" --source-video "${SOURCE_VIDEO}" \
    --output-dir "${OUTPUT_DIR}" --output-stem "${OUTPUT_STEM}" \
    --intensity-mode reference --mode "${MODE}" --mask-source global \
    --python "${PYTHON}" --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --block-mib "${BLOCK_MIB}" --ffprobe "${FFPROBE}" \
    --log-dir "${RUN_DIR}/details" "$@"
echo "Preparation completed. No training was launched."
