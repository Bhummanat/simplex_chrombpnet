# TanhSaturationMonitor Documentation
- Located in bpnet_model.py

## Purpose
`TanhSaturationMonitor` is a custom Keras callback designed to track the activation distributions of `tanh` layers during training. It helps diagnose whether the model is entering activation saturation, which can lead to:
- Vanishing gradients
- Slower learning
- Reduced representational power

## Why This Matters
The `tanh` activation squashes values into the range [-1, 1]. If too many inputs to `tanh` fall outside a moderate range (e.g. |x| > 2.5), gradients shrink drastically (near-zero slope), leading to poor learning.
BatchNorm can help mitigate this — but we want to measure how well it's working.

## How It Works
At the end of each epoch, this callback:
1. Passes a fixed dummy input through the model (forward-pass only).
2. Extracts the pre-activation inputs to all `tanh` layers.
3. Logs:
   - Mean, standard deviation, min, and max
   - % of values outside ±2.5 (an arbitrary but reasonable saturation threshold)

Additionally, it logs the output of the first convolution layer before BatchNorm.

## Dummy Input
- The dummy input is a fixed-encoded batch (1 or few samples).
- It is not used for training.
- It mimics the shape and format of real inputs (e.g., based on `one_hot`, `simplex_monomer`, or `simplex_dimer` encodings).

Its purpose is to act like a "test screw" for monitoring a machine: consistent, repeatable, not involved in training, but tells you how the system is behaving.

## Impact on Training Time
- Only runs once per epoch, during `on_epoch_end`.
- Involves forward pass only, so overhead is minimal.
- Can be tuned by adjusting the size of the dummy input.

## Example Output
Epoch 3: Activation Saturation Monitor (simplex_monomer)
  ➤ First Conv1D Output: mean=0.012, std=0.754, min=-2.95, max=2.80
  • `bpnet_1_activation` | mean=0.013, std=0.568, min=-0.98, max=0.98, 3.12% > ±2.5
  • `bpnet_2_activation` | mean=0.010, std=0.435, min=-0.94, max=0.95, 0.08% > ±2.5

## When to Use
Use if you're:
- Testing a new encoding method  
- Suspecting saturation issues  
- Curious about BatchNorm behavior with `tanh`  
- Comparing model variants (e.g., with vs. without BatchNorm)
