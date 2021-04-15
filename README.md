# Deep Learning - Project 2

## Performance of pre-trained CNNs (Deep-Transfer-Learning) as discriminator for Generative Neural Networks

This project researches the feasibility and performance of applying deep transfer learning on generative neural networks. This is done by taking an existing GAN architecture, and subsequently separating the generator from the discriminator. Then the discriminator is either left untrained, pre-trained on the desired dataset or pre-trained on a similar dataset (therefore applying deep transfer learning).

The following choices were made:
- The existing GAN architecture was chosen to be DCGAN ([implementation 1](https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py), [implementation 2](https://github.com/vwrs/dcgan-mnist/blob/master/model.py)).
- The [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/) and [Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist) datasets were used as both have $28\times 28 \times 1$ shaped images, are relatively small datasets and are both [contained in the Keras api](https://keras.io/api/datasets/).

## Structure of this repository

```
DL-PR2
│   
├─── discriminator1_digits
├─── discriminator1_fashion          
├─── discriminator2_digits          
├─── discriminator2_fashion
│           
└─── r_analysis   
```

The `discriminatorx_d` folders contain the pre-trained discriminators in `Tensorflow SavedModel` format. The 1 or 2 denotes whether the [first implementation](https://github.com/Leander-van-Boven/DL-PR2/blob/main/dcgan1.py) or [second implementation](https://github.com/Leander-van-Boven/DL-PR2/blob/main/dcgan2.py) was used. The name after the underscore denotes the dataset on which the discriminator was trained. The training method used can be found in the [`pretrain_disc.py`](https://github.com/Leander-van-Boven/DL-PR2/blob/main/pretrain_disc.py) file. Refer to [the TensorFlow documentation](https://www.tensorflow.org/guide/keras/save_and_serialize) on how to load these models.

The `r_analysis` folder contains the R scripts used to visualise the results of the experiments.

The multiple `.sh` files are used to execute the code on the Peregrine cluster, refer to the 'Running on Peregrine' section for more information.

`main.py` is the main entry point for the code, the experiments are executed by the code contained in `experiment.py`.

(`set_session.py` is a supplementary file that is used to initialise certain GPUs so that they can run TensorFlow)


## Running on Peregrine

> Peregrine is a compute cluster offered by the University of Groningen for its students. More information can be found [here](https://www.rug.nl/society-business/centre-for-information-technology/research/services/hpc/facilities/peregrine-hpc-cluster?lang=en).

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
