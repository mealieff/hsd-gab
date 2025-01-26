#!/bin/bash
#SBATCH -J HSD  # Job name
#SBATCH -p general                    # Partition name (use "general" or appropriate partition)
#SBATCH -o baseline_initial_%j.txt    # Standard output file with job ID
#SBATCH -e baseline_initial_%j.err    # Standard error file with job ID
#SBATCH --mail-type=ALL               # Email notifications for all job events
#SBATCH --mail-user=mealieff@iu.edu   # Email address for notifications
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --time=2-02:00:00             # Maximum run time (2 days, 2 hours)
#SBATCH --mem=16G                     # Memory allocation (16 GB)
#SBATCH -A r00018                     # SLURM account name


# Load the virtual environment - everything's been updated there
module load conda

# Activate the ENV
conda activate HSD

# Set up the working directory
cd ~/hsd-gab/

# Run the Python script
#python3 embedding-debugging.py
#python3 svm.py
#python3 get-embeddings.py   
#python3 svm-postembedding.py
#python3 svm-binary.py
#python3 svm-multi2.py
#python3 merged-resampler21.py
python3 merged-resampler31.py                        
