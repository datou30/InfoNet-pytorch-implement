import glob
import torch
import math
import time
import logging
import pickle
from sklearn.mixture import GaussianMixture
from scipy.stats import rankdata

from torch import nn
from torch import optim
from torch.optim import Adam
import numpy as np
import torch.autograd as autograd


from perceiver.encoder import PerceiverEncoder
from perceiver.decoder import PerceiverDecoder
from perceiver.perceiver import PerceiverIO
from perceiver.query import Query_Gen
from perceiver.query_new import Query_Gen_transformer, Query_Gen_transformer_PE
from util.epoch_timer import epoch_time, scale_data
from util.look_table import lookup_value_2d, lookup_value_close, lookup_value_average, lookup_value_bilinear, lookup_value_grid

from perceiver.encoder import PerceiverEncoder
from perceiver.decoder import PerceiverDecoder
from perceiver.perceiver import PerceiverIO
from perceiver.perceiver_lap import PerceiverIO_lap
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
from scipy.special import digamma
from scipy.spatial import KDTree
import scipy.special

latent_dim = 256
latent_num = 256
input_dim = 2
decoder_query_dim = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#batchsize = 32  #### num of distributions trained in one epoch
#seq_len = 5000  #### number of data points sampled from each distribution

encoder = PerceiverEncoder(
    input_dim=input_dim,
    latent_num=latent_num,
    latent_dim=latent_dim,
    cross_attn_heads=8,
    self_attn_heads=8,
    num_self_attn_per_block=8,
    num_self_attn_blocks=1
)

decoder = PerceiverDecoder(
    q_dim=decoder_query_dim,
    latent_dim=latent_dim,
)

query_gen = Query_Gen_transformer(
    input_dim = input_dim,
    dim = decoder_query_dim
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


model = PerceiverIO(encoder=encoder, decoder=decoder, query_gen = query_gen, decoder_query_dim = decoder_query_dim).cuda(device=0)
model = nn.DataParallel(model) #data parallel, make the model run on multiple GPUs
model.load_state_dict(torch.load('saved/model_2000_64_1000-42000--0.17.pt', map_location=device))
#
print(f'The model has {count_parameters(model):,} trainable parameters')

def infer(model, batch):
    model.eval()
    batch = torch.tensor(batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        mi_lb = model(batch)
    return mi_lb

def compute_smi_mean(sample_x, sample_y, model, proj_num):
    dx = sample_x.shape[1]
    dy = sample_y.shape[1]
    results = []
    seq_len = sample_x.shape[0]
    for i in range(proj_num//32):
        batch = np.zeros((32, seq_len, 2))
        ## 32 could be larger.
        for j in range(32):
            theta = np.random.randn(dx)
            phi = np.random.randn(dy)
            x_proj = np.dot(sample_x, theta)
            y_proj = np.dot(sample_y, phi)
            xy = np.column_stack((x_proj, y_proj))
            xy = scale_data(xy)
            batch[j, :, :] = xy
        infer1 = infer(model, batch).cpu().numpy()
        mean_infer1 = np.mean(infer1)
        results.append(mean_infer1)

    return np.mean(np.array(results))

if __name__ == '__main__':
    
    d = 10
    mu = np.zeros(d)
    sigma = np.eye(d)
    sample_x = np.random.multivariate_normal(mu, sigma, 1000)
    sample_y = np.random.multivariate_normal(mu, sigma, 1000)

    ## note that this checkpoint can suit any length data samples
    ## sample_x and sample_y has shape [seq_len, d_x] and [seq_len, d_y]
    ## this code uses sliced mutual information, and here the proj_num influence the result a lot, larger the better
    result = compute_smi_mean(sample_x, sample_y, model, proj_num=1000)
    print(result)