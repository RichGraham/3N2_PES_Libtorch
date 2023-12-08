#!/usr/bin/env python
#*********WARNING:********* Do not edit this directly, instead copy the file elsewhere and edit or find and edit the ipython file/Users/pmzrsg/Dropbox/Nitrogen_PES/3N2_Training/modQ_training/ 3N2_LHC_Learning.ipynb
#!/usr/bin/env python
# coding: utf-8
 
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
 
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
 
with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
    xRange = X[-4:-1]
    print(xRange)
    print(my_GP.predict(xRange))
 
class MeanVarModelWrapper(torch.nn.Module):
    def __init__(self, gp):
        super().__init__()
        self.gp = gp
    def forward(self, x):
        output_dist = self.gp(x)
        return output_dist.mean
 
model = my_GP.model
with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
    model.eval()
    test_x = X[-4:-1]
    pred = model(test_x)  # Do precomputation                                                                     \
                                                                                                                   
    traced_model = torch.jit.trace(MeanVarModelWrapper(model), test_x)
 
with torch.no_grad():
    traced_mean = traced_model(test_x)
 
print(torch.norm(traced_mean - pred.mean))
traced_model.save('traced_exact_gp.pt')
print( test_x)
print( traced_mean )
 
