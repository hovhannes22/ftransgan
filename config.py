import torch

# Model
lambda_content = 1
lambda_style = 1
lambda_l1 = 100
learning_rate = 2e-5

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4 # B
num_samples = 5 # K
epochs = 20
save_every = 5000
sample_every = 100

# Data
embedding_dim = 64
root_dir='./fonts/'
support_font = 'arial'