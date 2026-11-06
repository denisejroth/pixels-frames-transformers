# Module 05: Convolutional Neural Networks for Images

Instead of deciding in advance which features matter, a convolutional neural network (CNN)
learns them from labeled examples. This module opens up the architecture behind a decade of
computer vision: small filters that slide across the image, layers that build edges into shapes
and shapes into objects, and the training tricks that keep a network from memorizing its data.

## What you'll learn

- A short refresher on neural networks: neurons, layers, loss and gradient descent
- Convolutions as learned filters, applied by hand to a real image before a network learns them
- Pooling, feature maps, and how depth turns edges into parts and parts into objects
- Training a small CNN in PyTorch, with training and validation curves
- Overfitting, and the usual defenses: data augmentation, dropout and early stopping
- Looking inside a trained network: filters, activation maps, and saliency maps that show which
  pixels drove a prediction (and why those maps should be read with care)
- What CNNs need that social science data often lacks: thousands of labeled images

## Notebook

`05_cnns_for_images_exercises.ipynb` and `05_cnns_for_images_solutions.ipynb`

You first apply hand-written filters and max pooling to a portrait to see what a convolution
does. Then you build a small CNN in PyTorch, trace an image through its layers, and train it on
CIFAR-10, where 50,000 labeled images let it learn quickly. Trained on only 1,000 images, the same
network overfits, and you fight back with augmentation, dropout and early stopping. You look at
its learned filters, activation maps and saliency maps (and their limits), and finally train it on
the protest photos and score it on your 40 coded images, where a few hundred noisy labels turn
out not to be enough. On a CPU the notebook uses part of CIFAR-10 and fewer epochs.

## Slides

[`slides/05_cnns_for_images.pdf`](slides/05_cnns_for_images.pdf) (10 slides): designed versus
learned features, convolution and pooling, a small CNN layer by layer, the training loop,
overfitting and its defenses, looking inside a network, and why the protest task is hard for a CNN
trained from scratch.

The LaTeX source (`.tex`) and figures are in the same folder.

## Data and models

- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009) as a benchmark
  where data is plentiful, downloaded through torchvision
- The protest and King's Day photos with your codes from module 04, through the course helper
  (`shared/course_data.py`)
- PyTorch and torchvision; no pre-trained models

## Runtime

A GPU is recommended (*Runtime → Change runtime type → T4 GPU*). The notebook also runs on a CPU,
only more slowly.

## Builds on and leads to

Builds on module 04. Bonus modules 6 and 9 of
[Bags, Vectors & Transformers](https://github.com/denisejroth/bags-vectors-transformers/tree/main/bonus)
cover neural network basics and a first look at CNNs for images; this module goes further. Module
06 shows how to borrow a network that someone else trained on millions of images.

## Further reading

- Williams, N. W., Casas, A., & Wilkerson, J. D. (2020). *Images as data for social science
  research: An introduction to convolutional neural nets for image classification*. Cambridge
  University Press.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep
  convolutional neural networks. *Advances in Neural Information Processing Systems, 25*.
