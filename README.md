# GANobfuscator: Privacy-Preserving Synthetic Data Generation with GANs

GANobfuscator is a differentially private framework for generating high-quality synthetic data from sensitive datasets using Generative Adversarial Networks (GANs). It achieves this by incorporating differential privacy into the GAN training process through carefully adding noise to the model gradients. GANobfuscator allows leveraging the generative power of GANs while providing formal privacy guarantees within practical privacy budgets, without compromising data utility. Key features of it includes a gradient pruning strategy to improve training stability and scalability. 

# Dependencies:

- pytorch 
- torchmetrics
- torch-fidelity
- torchvision
- matplotlib
- scikit-learn
- imageio
- numpy
- tqdm
 






# Installation  guide:

```
# Install torch-fidelity (provides the metrics to evaluate GANs)
!pip install torch-fidelity

# Install PyTorchThis (provides the evaluation metric)
!pip install torchmetrics


# Install PyTorch
pip install torch torchvision

# Install tqdm (Progress bar library)
pip install tqdm

# Install imageio (Library for reading and writing image data)
pip install imageio

# Install scikit-learn (Machine Learning library)
pip install scikit-learn

# Install torchmetrics
pip install torchmetrics

# Install matplotlib (Plotting library)
pip install matplotlib

# Install numpy
pip install numpy

import warnings
```


# Running the file

- Download and open the code folder and then run the 'GANobfuscator_dp.ipynb file on Google Collab',

- Alternatively you can run this directly on collab with this link 


https://colab.research.google.com/drive/1miVuyKg9Y6mTSTuM8Qri4BFOtMzwShvo?usp=sharing


# Parameters used

| Parameter | Value | Description |
| --- | --- | --- |
| `workers` | 2 | Number of data loading threads |
| `batch_size` | 32 | Batch size for training |
| `nz` | 100 | Size of the latent noise vector |
| `ngf` | 64 | Number of feature channels in the generator |
| `ndf` | 64 | Number of feature channels in the discriminator |
| `beta1` | 0.5 | Beta1 parameter for Adam optimizer |
| `ngpu` | 1 | Number of GPUs to use for training |
| `num_test_samples` | 16 | Number of test samples |
| `nc` | 1 | Number of channels in the input data |
| `d_lr` | 5e-5 | Learning rate for the discriminator |
| `g_lr` | 5e-5 | Learning rate for the generator |
| `num_fake_images` | 1000 | Number of fake images to generate |
| `epsilon` | 10000,10,4| Epsilon value for differential privacy |
| `delta` | 1e-5 | Delta value for differential privacy |
| `num_epochs` | 15 | Number of training epochs |
| `max_per_sample_grad_norm` | 1e-2 | Maximum per-sample gradient norm |
| `max_grad_norm` | 1e-2 | Maximum gradient norm (same as `max_per_sample_grad_norm`) |

