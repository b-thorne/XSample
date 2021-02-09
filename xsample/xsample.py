import os

from absl import app
from absl import flags
from absl import logging

import gin

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
from tensorflow.keras.initializers import GlorotUniform as glorot_uniform

import matplotlib.pyplot as plt
import cosmoplotian.colormaps
from getdist import plots, MCSamples
import getdist

plt.rcParams['text.usetex'] = True

from pathlib import Path
import pickle

from datasets import LoadDataset

tfkl = tf.keras.layers
tfk = tf.keras
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

FLAGS = flags.FLAGS
PROJECT_DIR = Path(os.path.dirname(__file__))
RESULTS_DIR = "/global/cscratch1/sd/bthorne/dustvaeder/results"


@gin.configurable("IAF")
def IAF(dims=6, num_bijectors=5, hidden_units=[512, 512]):
    bijectors=[]
    for i in range(num_bijectors):
        iaf = tfb.Invert(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=tfb.AutoregressiveNetwork(params=2, hidden_units=hidden_units)))
        bijectors.append(iaf)
        bijectors.append(tfb.BatchNormalization())
        bijectors.append(tfb.Permute(permutation=list(range(dims))[::-1]))
    # Discard the last Permute layer.
    bijector = tfb.Chain(list(reversed(bijectors[:-1])))
    return bijector


@gin.configurable("MAF")
def MAF(dims=6, num_bijectors=5, hidden_units=[512, 512]):
    bijectors=[]
    for i in range(num_bijectors):
        bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=tfb.AutoregressiveNetwork(params=2, hidden_units=hidden_units)))
        bijectors.append(tfb.BatchNormalization())
        bijectors.append(tfb.Permute(permutation=list(range(dims))[::-1]))
    # Discard the last Permute layer.
    bijector = tfb.Chain(list(reversed(bijectors[:-1])))
    return bijector

@gin.configurable("BuildModel")
def BuildModel(dims, bijector, optimizer=tf.optimizers.Adam):
    q_x_z = tfd.TransformedDistribution(
        distribution=tfd.Sample(
            tfd.Normal(loc=0., scale=1.), sample_shape=[dims]),
        bijector=bijector)
    x = tfkl.Input(shape=[dims,], dtype=tf.float32)
    log_prob = q_x_z.log_prob(x)
    model = tfk.Model(x, log_prob)
    model.compile(optimizer=optimizer,
                loss=lambda _, log_prob: -log_prob)
    return model, q_x_z

@gin.configurable("Train", denylist=["train_dataset"])
def Train(train_dataset, model, batch_size=100, epochs=50, shuffle=True, verbose=True):
    (model, q_x_z) = model
    history = model.fit(x=train_dataset,
            y=np.zeros((train_dataset.shape[0], 0), dtype=np.float32),
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=train_dataset.shape[0] / batch_size,  
            shuffle=shuffle,
            verbose=verbose)
    return model, q_x_z, history

@gin.configurable("Dataset")
def Dataset(label):
    data = LoadDataset(label)
    return data

def Eval(model, q_x_z, training_dataset, results_dir):
    approximate_posterior = training_dataset.copy(label=r"$q_x(x)$")
    approximate_posterior.samples = q_x_z.sample(training_dataset.samples.shape[0]).numpy()

    g = plots.get_subplot_plotter(width_inch=6.5)
    g.triangle_plot([training_dataset, approximate_posterior], ['omegabh2', 'omegach2', 'theta', 'tau', 'omegak', 'logA', 'ns'], filled=True)
    g.export(str(results_dir / "densities" / "corner.pdf"))

    return 

def main(argv):
    del argv # unused
    gin.parse_config_file(FLAGS.gin_config)

    results_dir = Path(FLAGS.results_dir).absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    # read in dataset
    data = Dataset()
    # train model and evaluate
    if FLAGS.mode == "standard":
        model, q_x_z, history = Train(data.samples)
        Eval(model, q_x_z, data, results_dir)

    return

if __name__ == "__main__":
    flags.DEFINE_enum(
        "mode", 
        "standard", 
        ["standard"], 
        "Which mode to run in.")
    flags.DEFINE_string(
        "gin_config", 
        "./configs/config.gin", 
        "File containing the gin configuration.")
    flags.DEFINE_string(
        "results_dir", 
        "./results/default", 
        "Directory to write results.")
    app.run(main)