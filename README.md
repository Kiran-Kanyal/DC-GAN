# Generate Images Using DC-GAN | Generative Adversarial Networks

This project implements a Generative Adversarial Network (GAN) trained on the MNIST dataset. The goal is to generate new handwritten digits after training the model on real examples of handwritten digits.

---

## Work Flow

### 1. Data Loading
- Load and preprocess the MNIST dataset.
- Normalize the images to the range [-1, 1] to match the output of the generator's activation function.
- Create data loaders to efficiently feed batches of images into the network.

### 2. Model Architecture
- **Generator**: Takes random noise (latent vector) as input and outputs a 28x28 grayscale image using a series of transposed convolution layers to upsample the input.
- **Discriminator**: Takes a 28x28 image (real or generated) as input and outputs a probability score indicating whether the image is real or fake, using convolutional layers to downsample and classify.

### 3. Training Loop

#### Train the Discriminator:
- Use a batch of real images from the dataset.
- Generate fake images using the current Generator.
- Calculate the loss and update the Discriminator's weights.

#### Train the Generator:
- Generate fake images using random noise.
- Aim to fool the Discriminator, maximizing the probability that generated images are classified as real.
- Calculate the loss and update the Generator's weights.

*Repeat the above steps for a set number of epochs, saving model checkpoints and generated images periodically.*

### 4. Evaluation and Results
- Display sample generated images to visualize the quality of the model.
- Provide metrics or qualitative assessments of image realism.

---

## Tools and Frameworks
- **TensorFlow**: For building and training the GAN.
- **Matplotlib**: For visualizing results (generated images).
- **Pandas**: For handling data (if needed).
- **NumPy**: For numerical operations.
- **Seaborn**: For data visualization.
- **Jupyter**: For interactive development and model training.
- **Keras**: For building neural networks.



# Terminologies

### *Number of Epochs*
The number of times the entire training dataset is passed through the model.

---

### *Batch Size*
The number of samples processed before updating the model's parameters.

---

### *Noise*
A random input vector used by the generator in GANs to produce synthetic data.

---

### *Optimizer*
An algorithm that adjusts model weights to minimize the loss function.

---

### *lr (Learning Rate)*
A hyperparameter that controls the step size for updating model weights.

---

### *beta_1*
A parameter in the Adam optimizer that controls the decay rate of the first moment (gradient's mean).
