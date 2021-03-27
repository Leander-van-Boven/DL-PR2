#!/bin/bash

# Send an email when important events happen
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=<FILL OUT YOUR EMAIL ADDRESS HERE, not adding this to the git repo>

# Run for at most 4 hours
#SBATCH --time=04:00:00

# Run on v100, since they have the shortest queue times and to ensure 
# consistency between runs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

# Clean environment
module purge

# Load everything we need for TensorFlow (loads python, tensorflow and a lot more) and scikit
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4 scikit-learn/0.22.2.post1-fosscuda-2019b-Python-3.7.4 matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4

# Run the python script, outputting to a predefined output directory and passing any arguments passed to the bash file
python ~/deep_learning_course/project_2/main.py --log_dir ~/deep_learning_course/project_2/output/ $*