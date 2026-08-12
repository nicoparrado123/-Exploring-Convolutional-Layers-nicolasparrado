# Exploring Convolutional Layers - Fashion-MNIST

## Problem Description

This project looks at how convolutional layers change the way a neural network learns on image data.
The main focus is on understanding why convolutions work, what assumptions they bake into the model,
and how choices like kernel size, depth, and pooling actually affect results.

---

## Dataset - Fashion-MNIST

| Property        | Value                          |
|-----------------|-------------------------------|
| Source          | torchvision.datasets          |
| Samples         | 60000 train / 10000 test      |
| Image size      | 28x28 px, grayscale (1 ch)    |
| Classes         | 10 (T-shirt, Trouser, ...)    |
| Label balance   | Perfectly balanced (6000/cls) |

Fashion-MNIST works well for CNNs because clothing items are defined by local visual features like edges, stitching, and silhouettes. These are exactly the kind of patterns that convolutional filters are good at picking up. It's also harder than regular MNIST, which makes it a more interesting benchmark, and it's small enough to train on a laptop in a few minutes.

---

## Architecture Diagrams

### Baseline - Flatten + Dense

```
┌─────────────┐     ┌───────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│  Input      │     │  Flatten  │     │  Dense 256   │     │  Dense 128   │     │  Dense 10    │     │ Softmax  │
│  28x28      │────▶│  (784)    │────▶│  + ReLU      │────▶│  + ReLU      │────▶│              │────▶│          │
│  (1 ch)     │     │           │     │              │     │              │     │              │     │          │
└─────────────┘     └───────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────┘

Parameters: ~235K
```

### CNN (designed from scratch)

```
┌───────────┐   ┌─────────────────────┐   ┌──────────────┐   ┌─────────────────────┐   ┌──────────────┐
│  Input    │   │  Conv2d             │   │  MaxPool2d   │   │  Conv2d             │   │  MaxPool2d   │
│  1x28x28  │──▶│  1→16 ch, k=3 p=1  │──▶│  (2x2)       │──▶│  16→32 ch, k=3 p=1 │──▶│  (2x2)       │
│           │   │  + ReLU             │   │              │   │  + ReLU             │   │              │
└───────────┘   │  out: 16x28x28      │   │ out:16x14x14 │   │  out: 32x14x14      │   │ out:32x7x7   │
                └─────────────────────┘   └──────────────┘   └─────────────────────┘   └──────────────┘
                                                                                                │
                                                                                                ▼
                                          ┌──────────┐     ┌──────────────┐     ┌──────────────────────┐
                                          │ Softmax  │◀────│  Dense 10    │◀────│  Flatten (32x7x7=1568)│
                                          │          │     │              │     │  Dense 128 + ReLU     │
                                          └──────────┘     └──────────────┘     │  Dropout(0.3)         │
                                                                                 └──────────────────────┘

Parameters: ~234K (comparable to baseline for fair comparison)
```

Design justifications:
- **k=3, pad=1**: a 3x3 kernel is the smallest that captures structure in all directions. Padding keeps the spatial size intact before pooling so we don't lose information too early.
- **MaxPool(2)**: cuts the resolution in half after each conv block. This reduces the number of parameters and makes the model a bit more robust to small shifts in the image.
- **16 → 32 filters**: the first layer picks up simple things like edges and textures, the second one starts combining those into more complex shapes.
- **Dropout(0.3) only in the dense head**: adding dropout in the conv layers tends to hurt the spatial features, so I only regularize at the classification stage.

---

## Experimental Results

### Task 4 - Kernel Size Experiment (everything else fixed)

| Kernel | Test Acc | Params | Notes                          |
|--------|----------|--------|-------------------------------|
| 3x3    | ~91%     | ~234K  | Sharp local features, fast    |
| 5x5    | ~90%     | ~238K  | Slightly larger context       |
| 7x7    | ~89%     | ~248K  | Over-smooths fine textures    |

The 3x3 kernel came out on top. My guess is that the features that distinguish clothing classes (sole shape, stitching lines, straps) are small and detailed, so a larger kernel ends up blurring them together rather than capturing them cleanly.

### Baseline vs CNN

| Model    | Test Acc | Params |
|----------|----------|--------|
| Baseline | ~88%     | ~235K  |
| CNN 3x3  | ~91%     | ~234K  |

---

## Interpretation

**Why did the CNN outperform the baseline?**
The baseline flattens the image into a 784-element vector and loses all spatial information in the process. It has no way of knowing that pixel (3,4) is next to pixel (3,5). The CNN, on the other hand, looks at small patches of the image at a time, which matches how visual patterns actually work. On top of that, the same filter is applied across the whole image, so the model doesn't need to relearn the same edge detector for every position. With roughly the same number of parameters, the CNN just uses them more efficiently.

**What assumptions does convolution introduce?**
- Nearby pixels are more related than distant ones (locality)
- The same pattern can appear anywhere in the image (translation equivariance)
- Complex features are built from simpler ones layer by layer

**When would convolution not make sense?**
- Tabular data, where there's no spatial structure between columns
- Graph data, where connections are irregular
- Cases where long-range dependencies matter more than local patterns
- Very small datasets where a simple dense network is enough

---

## Repository Structure

```
.
|- README.md
|- notebook.ipynb
```
