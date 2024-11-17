# Generate images using DC-GAN | Generative Adversarial Networks
This project aims at implementing a generative adversarial network (GAN) trained on the MNSIT dataset. From this, we'll be able to generate new handwritten digits! We will train a GAN to generate new handwritten digits after showing it pictures of many real handwritten digits.

# Work Flow

1> Data Loading
   * Load and preprocess the MNIST dataset.
   * Normalize the images to the range [−1,1] to match the output of the generator's activation function.
   * Create data loaders to efficiently feed batches of images into the network.
     
2> Model Architecture
   * Generator: Takes random noise (latent vector) as input. Outputs a 28x28 grayscale image using a series of
     transposed convolution layers to upsample the input.
   * Discriminator: Takes a 28x28 image (real or generated) as input. Outputs a probability score indicating whether       the image is real or fake using convolutional layers to downsample and classify.
     
3> Training Loop
1. Train the Discriminator
      * Use a batch of real images from the dataset.
      * Generate fake images using the current Generator.
      * Calculate the loss and update the Discriminator weights.
2. Train the Generator
      * Generate fake images using random noise.
      * Aim to fool the Discriminator, maximizing the probability that generated images are classified as real.
      * Calculate the loss and update the Generator weights.
Repeat the above steps for a set number of epochs, saving model checkpoints and generated images periodically.
   
4> Evaluation and Results
   * Display sample generated images to visualize the quality of the model.
   * Provide metrics or qualitative assessment of image realism.
  


