# Exploring Convolutional Layers - Fashion-MNIST

## What this is about

This project explores how convolutional layers affect the way a neural network learns on image data.
The idea is to understand why convolutions work, what assumptions they make, and how decisions like
kernel size or pooling actually change the results.

---

## Dataset - Fashion-MNIST

- Source: torchvision.datasets
- Size: 60000 training images, 10000 test
- Images: 28x28 pixels, grayscale
- Classes: 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- Balanced: 6000 images per class

Fashion-MNIST is a good fit for CNNs because clothing items are defined by local visual features like
edges, stitching, and shape. That's exactly what convolutional filters are designed to pick up.
It's also harder than regular MNIST, which makes it more interesting to work with, and small enough
to train on a laptop in a few minutes.

---

## Architecture Diagrams

### Baseline: Flatten + Dense

```
Input 28x28 -> Flatten (784) -> Dense 256 + ReLU -> Dense 128 + ReLU -> Dense 10 -> Softmax

Parameters: around 235K
```

### CNN (designed from scratch)

```
Input 1x28x28
-> Conv2d 1->16 ch, kernel=3, pad=1 + ReLU  (output: 16x28x28)
-> MaxPool2d 2x2                             (output: 16x14x14)
-> Conv2d 16->32 ch, kernel=3, pad=1 + ReLU (output: 32x14x14)
-> MaxPool2d 2x2                             (output: 32x7x7)
-> Flatten (32x7x7 = 1568)
-> Dense 128 + ReLU
-> Dropout 0.3
-> Dense 10 -> Softmax

Parameters: around 207K
```

Design decisions:
- kernel=3, padding=1: 3x3 is the smallest kernel that captures structure in all directions. Padding keeps the spatial size before pooling so we don't lose info at the edges too early.
- MaxPool(2): halves the resolution after each conv block, reduces parameters and adds a bit of shift robustness.
- 16 then 32 filters: first layer looks for simple things like edges, second one starts combining those into more specific shapes.
- Dropout only in the dense head: putting dropout in the conv layers hurt performance, probably because it breaks up the spatial patterns the filters are learning.

---

## Results

### Task 4 - Kernel Size Experiment (everything else fixed)

| Kernel | Test Acc | Params |
|--------|----------|--------|
| 3x3    | around 91%     | around 207K  |
| 5x5    | around 92%     | around 215K  |
| 7x7    | around 92%     | around 228K  |

The differences are small. 3x3 is the lightest and still competitive. My guess is that for Fashion-MNIST,
where the important features are small and local (stitching lines, sole shape, straps), a smaller kernel
is enough. A larger one might help more on higher resolution images where structures are bigger.

### Baseline vs CNN

| Model    | Test Acc | Params |
|----------|----------|--------|
| Baseline | around 88%     | around 235K  |
| CNN 3x3  | around 91%     | around 207K  |

---

## Interpretation

**Why did the CNN outperform the baseline?**

When you flatten a 28x28 image into a 784-element vector, you lose all spatial information.
The model has no idea that pixel (3,4) is next to pixel (3,5). The CNN avoids this by looking
at small local patches instead of the whole image at once. The same filter slides across the
entire image, so if it learns to detect a horizontal edge somewhere, it can detect it anywhere.
With fewer parameters, the CNN just uses them more efficiently.

**What assumptions does convolution introduce?**

- Nearby pixels are more related than distant ones (locality)
- The same pattern can appear anywhere in the image (translation equivariance)
- Complex features are built from simpler ones layer by layer

**When would convolution not make sense?**

- Tabular data: columns have no spatial relationship, sliding a filter over them doesn't mean anything
- Graph data: connections are irregular and don't fit a grid
- When long-range dependencies matter more than local patterns
- Very small datasets where a simple dense network is enough

---

## Repo structure

```
.
|- README.md
|- notebook.ipynb
```
