# Radio interferometric calibration with PyTorch
This is a simple example to compare popular optimizers used in deep learning (Adam etc.) with stochastic LBFGS.

The stochastic LBFGS optimizer is provided with the code. Further details are given [in this paper](https://ieeexplore.ieee.org/document/8755567) and [also this](https://arxiv.org/abs/2003.00986). Also see [this introduction](http://sagecal.sourceforge.net/pytorch/index.html).

Files included are:

``` lbfgsnew.py ```: New LBFGS optimizer

``` run_calibration.py ```: Run a simple calibration

<img src="time.png" alt="reduction of calibration cost" width="700"/>

Here is an image showing the reduction of calibration error (Student's T loss) with minibatch (CPU time) for LBFGS and Adam. Adam runs faster but slower to converge. LBFGS uses 1 epoch and Adam uses 4 epochs in the image. The minibatch size is 1/10-th of the full dataset.

For a much faster, C/CUDA version of the LBFGS optimizer, follow [this link](https://github.com/nlesc-dirac/sagecal/tree/master/test/Dirac).

For a completely different method to calibrate, see also [ManOpt](https://github.com/NicolasBoumal/manopt/blob/master/examples/radio_interferometric_calibration.m).
