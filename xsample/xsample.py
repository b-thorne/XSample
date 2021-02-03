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

from pathlib import Path
import pickle

from datasets import load_dataset

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
def Dataset(label, nsamples, dims):
    data, info = load_dataset(label)
    # return just the six LCDM parameters for now.
    info['nsamples'] = nsamples
    return data[:nsamples, 2:2+dims], info

def Eval(model, q_x_z, results_dir):
    training_data, training_info = Dataset()
    full_data, full_info = Dataset(nsamples=-1)
    training_samples = MCSamples(samples=training_data, label=f"PlikHM TTTEEE+lowl+lowE, training subset of {training_info['nsamples']}")
    prior_samples = MCSamples(samples=q_x_z.sample(full_data.shape[0]).numpy(), label="$q(x | z)$")
    data_samples = MCSamples(samples=full_data, label=f"PlikHM TTTEEE+lowl+lowE, full dataset {full_info['nsamples']}")
    g = plots.get_subplot_plotter()
    g.triangle_plot([training_samples, prior_samples, data_samples], filled=True)
    g.export(str(results_dir / "densities" / "corner.pdf"))

    g = plots.get_subplot_plotter()
    pulled_back_data = MCSamples(samples=q_x_z.bijector.inverse(full_data).numpy(), label="PlikHM TTTEEE+lowl+lowE, pulled back")
    g.triangle_plot(pulled_back_data, filled=True)
    g.export(str(results_dir / "densities" / "pulled_back.pdf"))
    return 

def main(argv):
    del argv # unused
    gin.parse_config_file(FLAGS.gin_config)
    
    results_dir = Path(FLAGS.results_dir).absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    # read in dataset
    data, info = Dataset()

    # train model and evaluate
    if FLAGS.mode == "standard":
        model, q_x_z, history = Train(data)
        Eval(model, q_x_z, results_dir)

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