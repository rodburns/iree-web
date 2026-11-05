---
hide:
  - tags
tags:
  - CPU
icon: octicons/cpu-16
---

# Quickstart on CPU

This guide walks you through compiling a model with IREE and running it on
your CPU. IREE uses [LLVM](https://llvm.org/) to compile the dense
computations in each program into highly optimized native code for your CPU.

By default, IREE includes these LLVM CPU targets:

* X86
* ARM
* AArch64
* RISCV

Other targets may work, but in-tree test coverage and performance work is
focused on that list.

At runtime, compiled CPU programs run on one of IREE's CPU devices:

* `local-task`: asynchronous, multithreaded device built on IREE's "task"
   system
* `local-sync`: synchronous, single-threaded device that executes work inline

## :octicons-download-16: Prerequisites

### Get the IREE compiler

#### :octicons-download-16: Download the compiler from a release

Python packages are distributed through multiple channels. See the
[Python Bindings](../../reference/bindings/python.md) page for more details.
The core [`iree-base-compiler`](https://pypi.org/project/iree-base-compiler/)
package includes the compiler tools:

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-compiler-from-release.md"

### Get the IREE runtime

You will need an IREE runtime that supports the local CPU HAL driver.

#### :octicons-download-16: Download the runtime from a release

Python packages are distributed through multiple channels. See the
[Python Bindings](../../reference/bindings/python.md) page for more details.
The core [`iree-base-runtime`](https://pypi.org/project/iree-base-runtime/)
package includes the local CPU HAL drivers:

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-runtime-from-release.md"

#### :octicons-checklist-24: Check for CPU devices

You can check for CPU support by looking for the `local-sync` and `local-task`
drivers and devices:

```shell
iree-run-module --list_drivers
```

```console hl_lines="9-10"
--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-run-module-driver-list.md:2"
```

```shell
iree-run-module --list_devices
```

```console hl_lines="3-4"
--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-run-module-device-list-amd.md"
```

## Compile and run a program

With the requirements out of the way, we can now compile a model and run it.

### :octicons-file-code-16: Import a model

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-import-onnx-mobilenet.md"

### :octicons-file-code-16: Compile a program

Compile the model for your CPU with the `local` device and the `llvm-cpu`
compiler backend:

```shell
iree-compile \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    mobilenetv2.mlir -o mobilenet_cpu.vmfb
```

This uses the default settings, which produce a program that runs on any CPU
with the same architecture as your machine. You'll make it faster for your CPU
[after you've seen it working](#make-it-faster-for-your-cpu).

### :octicons-terminal-16: Classify an image

MobileNetV2 is an image classifier. Given a photo, it scores how likely the
photo is to show each of the 1,000 object classes in the
[ImageNet](https://www.image-net.org/) dataset. In this section you will use
the compiled model to identify the breed of a dog in a photo.

#### Install the helper packages and class labels

Install [NumPy](https://numpy.org/) and [Pillow](https://python-pillow.org/),
which the scripts below use to prepare the image and read the results:

```shell
python3 -m pip install numpy pillow
```

Download the list of ImageNet class names. Line *n* of this file is the name
of the class that the model scores at position *n* of its output:

```shell
curl -L -o imagenet_classes.txt https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

#### Download a photo

Download a photo of a dog from Wikipedia:

```shell
curl -L -o dog.jpg https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg
```

![A yellow Labrador retriever](https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg){width="300"}

#### Convert the photo to the model's input format

The model doesn't accept a JPEG file. It expects a 1x3x224x224 tensor of
32-bit floats:

* `1`: a batch of one image.
* `3`: the red, green and blue color channels, each stored as a separate
  224x224 plane.
* `224x224`: the image size in pixels.

Each value must also be normalized in the same way as the images the model
was trained on. Save the following script as `preprocess.py`. It resizes and
crops the photo, normalizes the pixel values, and saves the result as a NumPy
file:

```python title="preprocess.py"
import sys

import numpy as np
from PIL import Image

# Load the photo and make sure it has three color channels.
image = Image.open(sys.argv[1]).convert("RGB")

# Resize so the shorter side is 256 pixels, then crop the center 224x224.
width, height = image.size
scale = 256 / min(width, height)
image = image.resize((round(width * scale), round(height * scale)), Image.BILINEAR)
width, height = image.size
left, top = (width - 224) // 2, (height - 224) // 2
image = image.crop((left, top, left + 224, top + 224))

# Scale pixels to [0, 1] and normalize with the ImageNet mean and std.
pixels = np.asarray(image, dtype=np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
pixels = (pixels - mean) / std

# Reorder from HWC to NCHW and add a batch dimension: 1x3x224x224.
pixels = pixels.transpose(2, 0, 1)[np.newaxis, ...]
np.save("input.npy", pixels.astype(np.float32))
print(f"Saved input.npy with shape {pixels.shape}")
```

Run it on the photo:

```shell
python3 preprocess.py dog.jpg
```

```text
Saved input.npy with shape (1, 3, 224, 224)
```

#### Run the model

Run the compiled model on the CPU with `iree-run-module`, reading the input
from `input.npy` and writing the scores to `output.npy`:

``` shell hl_lines="2 5 6"
iree-run-module \
    --device=local-task \
    --module=mobilenet_cpu.vmfb \
    --function=torch-jit-export \
    --input=@input.npy \
    --output=@output.npy
```

* `--device=local-task` runs the model on all CPU cores. Use `local-sync` to
  run it on a single thread.
* `--function=torch-jit-export` is the name of the model's entry point, which
  the ONNX importer took from the original model.
* `--input=@input.npy` and `--output=@output.npy` read the input tensor from,
  and write the output tensor to, NumPy files. The `@` prefix tells
  `iree-run-module` that the value is a file name.

#### Read the results

The output is a 1x1000 tensor with one score for each ImageNet class. Save the
following script as `postprocess.py`. It converts the scores to probabilities
and prints the five most likely classes:

```python title="postprocess.py"
import numpy as np

# Load the 1x1000 scores written by iree-run-module.
logits = np.load("output.npy")[0]

# Convert the scores to probabilities.
probabilities = np.exp(logits - logits.max())
probabilities /= probabilities.sum()

with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f]

# Print the five most likely classes.
for index in np.argsort(probabilities)[::-1][:5]:
    print(f"{probabilities[index]:6.2%}  {labels[index]}")
```

Run it:

```shell
python3 postprocess.py
```

The model correctly identifies the dog as a Labrador retriever:

```text
53.49%  Labrador retriever
15.25%  Ibizan hound
 4.48%  Weimaraner
 3.26%  Saluki
 2.97%  golden retriever
```

The exact percentages may differ slightly depending on your CPU and compiler
version.

Try it with your own photos by running `preprocess.py`, `iree-run-module` and
`postprocess.py` again with a different image.

## Make it faster for your CPU

The program you compiled above uses a generic instruction set, so it runs on
any CPU of the same architecture. To also use the faster vector and matrix
instructions that your CPU supports, such as AVX-512 on x86 or SVE on Arm,
recompile with `--iree-llvmcpu-target-cpu=host`:

``` shell hl_lines="4"
iree-compile \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    mobilenetv2.mlir -o mobilenet_cpu_host.vmfb
```

Run `iree-run-module` again with `--module=mobilenet_cpu_host.vmfb` and you'll
get the same classification. To measure the speed-up, run
`iree-benchmark-module` with each `.vmfb` file and compare the `Time` column:

```shell
iree-benchmark-module \
    --device=local-task \
    --module=mobilenet_cpu_host.vmfb \
    --function=torch-jit-export \
    --input=@input.npy
```

!!! warning

    A program compiled with `host` only runs on CPUs that support the same
    instructions as the machine that compiled it. To compile for a different
    machine or architecture, see
    [Choosing CPU targets](../deployment-configurations/cpu.md#choosing-cpu-targets).

## Next steps

* See the [CPU deployment guide](../deployment-configurations/cpu.md) for
  optimization flags, cross-compiling and other deployment options.
