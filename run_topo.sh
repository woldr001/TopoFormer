#!/bin/bash --login
#SBATCH --job-name=topo_1a4k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=topo_1a4k_%j.out

# Use the directory where sbatch was called from (avoids ~/home mount path issues)
cd "$SLURM_SUBMIT_DIR"

# Initialize conda shell hooks so that 'conda activate' works in batch jobs
module load Conda/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate topoformer

mkdir -p ./output

python code_pkg/main_potein_ligand_topo_embedding.py \
  --output_feature_folder "./output" \
  --protein_file "/mnt/home/woldring/TopoFormer/1a4k_protein_mono.pdb" \
  --ligand_file  "/mnt/home/woldring/TopoFormer/1a4k_ligand.mol2" \
  --ligand_file_type "mol2" \
  --dis_start 0 --dis_cutoff 10 --dis_step 0.1 \
  --consider_field 20
