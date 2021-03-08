# Deep Learning - Project 2

## Generalisability of ImageNet CNN's as discriminator network in GAN models

## Running on Peregrine

You can run the script on Peregrine as follows:

```sh
sbatch --job-name=JOBNAME main.sh --you "can" --use "python" --args "here"
```

Make sure to supply a descriptive job name, otherwise it will be impossible to
find your results. Also check the run time when a job finishes and adjust the
sbatch `time` parameter accordingly to avoid allocating too much time. For
convenience, the output location is provided in the bash file by default.

The bash file currently assumes the project files are located at
`/home/$USER/deep_learning_course/project_2/`, and will write its output to
`/home/$USER/deep_learning_course/project_2/output/` (as long as the python code
respects the `--outdir` argument).
