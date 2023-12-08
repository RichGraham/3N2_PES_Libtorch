#!/usr/bin/env python

import sys,os,os.path
sys.path.append(os.path.expanduser('~/source/PES_Torch/') )
sys.path.append("/home/p/pmzrsg/source/PES_Torch/..")
import torch
import gpytorch
import PES_Torch
import numpy as np
np.set_printoptions(precision=16)
torch.set_printoptions(precision=50)
my_likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Interval(1e-19,3e12),
    noise_prior=gpytorch.priors.NormalPrior(0, 1e-15)
)
my_gpytorch_model = PES_Torch.DefaultTorch_GPModel(kernel_function=PES_Torch.SymmetricRBFKernel_3Linears,
                                                likelihood = my_likelihood,
                                                lengthscale_prior = gpytorch.priors.NormalPrior(0.1, 0.03)
                                                )
my_gpytorch_model.mean_module = gpytorch.means.ZeroMean()
my_GP = PES_Torch.GpyTorch_WithFixedIteration(my_gpytorch_model, training_iter=0)

hypers = {
    'likelihood.noise_covar.noise': torch.tensor(7.e-08 ),
    'covar_module.outputscale': torch.tensor(4.e-09),
    'covar_module.base_kernel.lengthscale': torch.tensor(0.7 ),
}
my_GP.set_hyper_params( hypers )

X,y,_ = PES_Torch.loadfile_3body_withIndices('3N2_i1e5_p128_g9900-nested.lhc')

end=10
my_GP.train(X[:end], y[:end])

xRange = X[-4:-1]
print(my_GP.predict(xRange))
print(xRange)

