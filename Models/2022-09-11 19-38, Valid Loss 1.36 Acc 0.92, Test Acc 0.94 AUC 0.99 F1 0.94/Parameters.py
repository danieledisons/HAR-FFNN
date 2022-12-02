#PyTorch v1.11.0
#CUDA device available: True
#1 devices available
#device = cuda
#isnotebook = True
#isgooglecolab = False
#shell = ZMQInteractiveShell

layer_type = ["dense", "dense", "dense"]
Seed = 42
num_units = [1200, 128, 256, 512]
activation = ["relu", "relu", "relu"]
dropout = [0.3, 0.3, 0.3]
usebias = [True, True, True, True]
batch_size = 256
K_Length = 8
D_Length = 1200
H1 = 1
W1 = 1
conv_input_size = (1200,)
input_size = 1200
output_size = 1
hn1 = 1200

l2_lamda = 0.25
mu = 0.99
batchnorm_momentum = [None, None, None]
conv_pool_size = None
conv_pool_stride = None
conv_pool_padding = None
conv_pool_dilation = None

PrintInfoEverynEpochs = 1

train_best_loss = 0.03280576691031456
valid_best_loss = 1.353424310684204
ReluAlpha = 0
EluAlpha = 0.8
valid_metric1 = 0.917184265010352
valid_metric2 = 0.980255917188077
valid_metric3 = 0.9167280676770806
valid_best_metric1 = 0.9296066252587992
valid_best_metric2 = 0.9818145390870582
valid_best_metric3 = 0.9290507462660497
