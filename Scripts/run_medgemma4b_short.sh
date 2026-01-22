#!/bin/bash
#SBATCH --job-name=run_medgemma4b_short_diagnose     # Job name
#SBATCH --output=medgemma4b_short_diagnose_%j.out
#SBATCH --error=medgemma4b_short_diagnose_%j.err
#SBATCH --time=22:00:00           # Time limit hrs:min:sec
#SBATCH --account=def-zshakeri
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --tmp=200G
#SBATCH --mem=80G
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=8

module load gcc arrow/21.0.0 python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
module load gcc arrow/21.0.0 python/3.11
module spider rust
module load rust
module load gcc arrow/21.0.0 python/3.11
module load cuda torch
pip install --no-index transformers datasets evaluate accelerate bitsandbytes unsloth


# Choose a writable cache/tmp location with fallbacks.
CANDIDATES=("${SLURM_TMPDIR:-}" "/scratch/${SLURM_JOB_ID:-unknown}" "$HOME/.cache/huggingface" "/tmp/${USER}_hf_cache" "$PWD/.cache")
CACHE_DIR=""
for d in "${CANDIDATES[@]}"; do
	if [ -z "$d" ]; then
		continue
	fi
	mkdir -p "$d" 2>/dev/null || true
	if [ -d "$d" ] && [ -w "$d" ]; then
		CACHE_DIR="$d"
		break
	fi
done

if [ -z "$CACHE_DIR" ]; then
	echo "ERROR: No writable cache directory found. Please create a writable folder and set HF_HOME or TRANSFORMERS_CACHE." >&2
	exit 1
fi

export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export XDG_CACHE_HOME="$CACHE_DIR"
export TMPDIR="$CACHE_DIR/tmp"
mkdir -p "$TMPDIR" || true

echo "Using cache dir: $CACHE_DIR"

python /home/haji80as/EMBC_project/medgemma4b_short_diagnose.py