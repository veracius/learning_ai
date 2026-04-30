# nn.h

Header-only multi-layer perceptron built on top of `ugrad.h`'s scalar autograd. Each weight, activation, and intermediate computation is a `Value`; the entire network plus loss is one big graph that `Forward` and `Backward` traverse.

## Files

- `ugrad.h` — autograd primitives: `Value`, op constructors (`AddValues`, `MulValues`, `PowValues`, `TanhValue`, `SubValues`, ...), `Forward`/`Backward`/`ZeroGrad`, topo helpers.
- `nn.h` — `Neuron`, `Layer`, `MLP`, `Trainer`, plus orchestration helpers `CreateTrainerSystem` / `FreeTrainerSystem`.
- `nn_test.c` — end-to-end smoke test: trains `MLP(3, [4, 4, 1])` on Karpathy's 4-sample dataset.

## Data model

```
Neuron      weights[]  features[]  products[]  out (tanh)
Layer       neurons[]  out[] (alias of each neuron's out)
MLP         layers[]   out[] (alias of last layer's out)
Trainer     model      loss_out (scalar)  targets[]  topo (cached)  learning_rate
```

`Neuron.out` is `tanh(sum(weights[i] * features[i]))`. `Layer.out[k]` aliases `neurons[k]->out`. `MLP.out[k]` aliases `layers[last]->out[k]`.

## Bias convention

`features[0]` is the bias slot. It is created as a `Value(1.0)` by `CreateNeuron` and stays at 1.0 forever; `weights[0]` is the learnable bias. `features[1..num_features-1]` are real inputs. So a neuron that takes K inputs has `num_features == K + 1`.

`CreateMLP(n_features, n_neurons_per_layer, n_layers)` already handles this: layer 0 neurons are sized `n_features + 1`, layer `i+1` neurons are sized `n_neurons_per_layer[i] + 1`.

## Ownership

- `Neuron.owns_features` is 1 after `CreateNeuron` (the dummy 1.0 features it allocated). It flips to 0 after `ConnectNeuronFeatures`/`ConnectLayers`/`ConnectMLPInputs` rewires features[1..] to externally-owned Values. `features[0]` (bias) is always neuron-owned regardless.
- The MLP's output array (`mlp->out`) aliases the last layer's `out[]`; `FreeMLP` does not free it separately.
- `FreeNeuron` walks the sum chain via `out->prev[0]` to free the Add nodes between `out` and `products[0]`.
- `FreeTrainer` uses set-difference: walk `trainer->topo`, free anything not reachable from `model->out` (those are the loss-only nodes — Sub-Adds, Neg-Muls, orphan -1.0 / 2.0 leaves, Pow heads, sum-chain Adds across outputs, target leaves).

## Typical workflow

```c
#include "nn.h"

unsigned int layers[] = {4, 4, 1};
Trainer *t = CreateTrainerSystem(/*n_features=*/3, layers, /*n_layers=*/3, MSELoss);
t->learning_rate = 0.05;

for (int epoch = 0; epoch < N; epoch++) {
    for (int s = 0; s < n_samples; s++) {
        SetMLPInputs(t->model, xs[s], 3);
        SetTrainerTargets(t, &ys[s], 1);

        ZeroGrad (t->loss_out, t->topo);
        Forward  (t->loss_out, t->topo);
        Backward (t->loss_out, t->topo);
        MLPStep  (t);
    }
}

FreeTrainerSystem(t);
```

After `Forward`, current predictions are at `t->model->out[k]->data` and the combined scalar loss is at `t->loss_out->data`.

## API surface

### MLP

| Function | Purpose |
| --- | --- |
| `CreateMLP(n_features, n_per_layer[], n_layers)` | Build the static graph; layers connected, layer 0 has dummy features. |
| `ConnectMLPInputs(mlp, inputs, count)` | Replace layer-0 dummies with caller-owned `Value*`s. Must run **before** `CreateTrainer`. |
| `SetMLPInputs(mlp, double *xs, count)` | Write per-sample input values into layer 0's `.data` fields. |
| `MLPParameters(mlp, *out_count)` | Returns a malloc'd `Value**` of every weight (bias as `weights[0]`). Caller frees the array. |
| `FreeMLP(mlp)` | Free MLP-owned Values; leaves caller-supplied input Values alone. |

### Trainer

| Function | Purpose |
| --- | --- |
| `CreateTrainer(model, loss_func)` | Allocate Trainer, call `loss_func(t)` to populate `targets[k]` and `loss_out`, cache topo from `loss_out`. Default `learning_rate = 0.05`. |
| `MSELoss(t)` | Standard loss builder: sets `loss_out = sum_k (out[k] - target[k])^2`. Pass as `loss_func` to `CreateTrainer`. |
| `SetTrainerTargets(t, double *ys, count)` | Write per-sample targets into `targets[k]->data`. |
| `MLPStep(t)` | Vanilla SGD: `w->data -= t->learning_rate * w->grad` for every weight. |
| `FreeTrainer(t)` | Free loss-only Values via set-difference vs `model->out`. Must run **before** `FreeMLP`. |

### TrainerSystem (orchestration)

| Function | Purpose |
| --- | --- |
| `CreateTrainerSystem(n_features, n_per_layer[], n_layers, loss_func)` | Runs `CreateMLP` → allocate inputs → `ConnectMLPInputs` → `CreateTrainer` in the correct order. Returns `Trainer*`. |
| `FreeTrainerSystem(t)` | Snapshots input pointers from layer 0, then `FreeTrainer` → `FreeMLP` → free inputs. Use this instead of calling the individual frees. |

### ugrad traversal (used directly)

| Function | Purpose |
| --- | --- |
| `Forward(root, topo)` | Walk forward; `topo=NULL` rebuilds internally, otherwise reuses the passed topo. |
| `Backward(root, topo)` | Same convention. Sets `root->grad = 1.0` and walks reverse topo; intermediate backwards accumulate via `+=`. |
| `ZeroGrad(root, topo)` | Same convention. Zeroes every grad in the topo. |

## Critical rules

1. **Wire inputs before building the Trainer.** `CreateTrainer` caches a topo built from `loss_out`. If `ConnectMLPInputs` runs after that, the topo references the old dummy pointers (or freed memory). `CreateTrainerSystem` enforces the right order.
2. **Update by `.data`, not by re-pointering.** Once the topo is cached, do not call `ConnectMLPInputs` again. Use `SetMLPInputs` to overwrite `.data` between samples — the graph topology stays fixed.
3. **Free in reverse build order.** `FreeTrainer` before `FreeMLP` (FreeTrainer's set-difference needs the MLP graph alive). `FreeTrainerSystem` handles this.
4. **Online SGD per sample.** `ZeroGrad` between samples is required; without it, residual grads on intermediate nodes get re-propagated through the chain rule when their backwards run, double-counting.

## Build and run the test

```
gcc nn_test.c -lm -o nn_test
./nn_test
```

Expected output: loss decreasing monotonically over 20 epochs (≈4.2 → ≈0.06), final predictions within ~0.16 of targets.
