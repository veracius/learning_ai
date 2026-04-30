#ifndef NN_H
#define NN_H

#include "ugrad.h"

//Uses ugrad Value data structs to implement the math
//Neuron, Layer, MLP, are each just a giant tree of Value structs with a single output: the root node
//Use Forward/Backward Functions from the Neuron Value *out member to computer prediction/backprop

typedef struct Neuron{
   unsigned int num_features;
   Value **weights; //length num_features includes the bias as weights_0
   Value **features;
   Value **products;
   Value *out;
   int owns_features; //1 if features[] were allocated by CreateNeuron; 0 after ConnectNeuronFeatures rewires them to values owned by another neuron
}Neuron;

typedef struct Layer{
   unsigned int num_neurons;
   Neuron **neurons;
   Value **out;
}Layer;

typedef struct MLP{
   unsigned int num_layers;
   Layer **layers;
   unsigned int num_out;
   Value **out;
}MLP;

typedef struct Trainer{
   MLP *model;
   unsigned int num_out;
   Value *loss_out;       //single combined loss root, e.g. sum_k (pred_k - target_k)^2 for MSE
   Value **targets;       //per-output target Values, length num_out; caller updates targets[k]->data per sample
   ValueList *topo;       //cached topo from loss_out; valid only while inputs/topology are not re-pointered
   double learning_rate;  //SGD step size used by MLPStep; CreateTrainer initialises to 0.05; caller may override
}Trainer;

//=============================================================================
//Neuron Functions
//=============================================================================
Neuron* CreateNeuron(unsigned int nf){
   Neuron *out         = malloc(sizeof(Neuron));
   out->num_features   = nf;
   out->weights        = malloc(nf*sizeof(Value *));
   for(int i=0; i<nf; i++) out->weights[i] = CreateValue(((double)rand() / RAND_MAX) * 2.0 - 1.0);
   
   //Create the Value data tree for the Neuron itself
   out->features       = malloc(nf*sizeof(Value *));
   out->products       = malloc(nf*sizeof(Value *));
   for(int i=0; i<nf; i++){
      out->features[i] = CreateValue(1.0); //All demo features are 1.0f to start off; x_bias should stay 1.0f
      out->products[i] = MulValues(out->weights[i], out->features[i]);
   }
   Value *sum          = SumValues(out->products, nf);
   out->out            = TanhValue(sum);
   out->owns_features  = 1;

   return out;
}

void SetNeuronWeights(Neuron *n, unsigned int count, float *weights){
   if(NULL == n || NULL == weights || count != n->num_features) return;
   for(unsigned int i = 0; i < count; i++) n->weights[i]->data = (double)weights[i];

   return;
}

//v must contain (num_features - 1) inputs; index 0 is the bias slot and is always owned by the neuron, so it's skipped here
void ConnectNeuronFeatures(Neuron *n, Value **v, unsigned int count){
   if(NULL == n || NULL == v || count != n->num_features - 1) return;

   for(unsigned int i = 1; i < n->num_features; i++){
      if(n->owns_features) FreeValue(n->features[i]); //only free the dummies CreateNeuron made; borrowed features are owned by another neuron
      n->features[i]          = v[i-1];
      n->products[i]->prev[1] = v[i-1];
   }
   n->owns_features = 0;

   return;
}

//Frees every Value owned by the neuron: tanh output, sum chain, products, weights, bias dummy at features[0], and the other feature dummies only if still owned.
//Does not free Values borrowed from other neurons or from user-supplied inputs.
void FreeNeuron(Neuron *n){
   if(NULL == n) return;

   //Free the output tanh node; grab the sum root via prev[0] before the tanh is gone
   Value *sum_root = n->out->prev[0];
   FreeValue(n->out);

   //Free the Add chain created by SumValues. The innermost Add's prev[0] is products[0] (the stop condition).
   //If num_features == 1, SumValues returned products[0] directly so sum_root == products[0] and the loop never enters.
   Value *node = sum_root;
   while(node != n->products[0]){
      Value *next = node->prev[0];
      FreeValue(node);
      node = next;
   }

   //Free the per-feature Value structs (Mul product nodes, weight leaves)
   for(unsigned int i = 0; i < n->num_features; i++){
      FreeValue(n->products[i]);
      FreeValue(n->weights[i]);
   }

   //Bias at features[0] is always owned by the neuron; the rest only if never rewired
   FreeValue(n->features[0]);
   if(n->owns_features)
      for(unsigned int i = 1; i < n->num_features; i++)
         FreeValue(n->features[i]);

   free(n->weights);
   free(n->features);
   free(n->products);
   free(n);

   return;
}

//=============================================================================
//Layer Functions
//=============================================================================

Layer* CreateLayer(unsigned int n_features, unsigned int n_neurons){
   Layer *neuronlayer         = malloc(sizeof(Layer));

   neuronlayer->num_neurons   = n_neurons;
   neuronlayer->neurons       = malloc(n_neurons*sizeof(Neuron *));
   neuronlayer->out           = malloc(n_neurons*sizeof(Value *));
   for(int i=0; i<n_neurons; i++){
      neuronlayer->neurons[i] = CreateNeuron(n_features);
      neuronlayer->out[i]     = neuronlayer->neurons[i]->out;
   }

   return neuronlayer;
}

//Wires every neuron in l2 to take l1's outputs as its inputs (features[1..]); l2's neurons must have num_features == l1->num_neurons + 1
void ConnectLayers(Layer *l1, Layer *l2){
   if(NULL == l1 || NULL == l2) return;

   for(unsigned int j = 0; j < l2->num_neurons; j++)
      ConnectNeuronFeatures(l2->neurons[j], l1->out, l1->num_neurons);

   return;
}

void FreeLayer(Layer *l){
   if(NULL == l) return;

   for(unsigned int j = 0; j < l->num_neurons; j++)
      FreeNeuron(l->neurons[j]);

   free(l->neurons);
   free(l->out); //just the Value* array; the underlying Value structs were freed by FreeNeuron via n->out
   free(l);

   return;
}

//=============================================================================
//MLP & Trainer Functions
//=============================================================================

//The outputs of all the neurons in one layer form the features for each neuron in the next layer,
//except in the case of a single neuron (which the last layer should be), which outputs a final value
MLP* CreateMLP(unsigned int n_features, unsigned int n_neurons_per_layer[], unsigned int n_layers){
   if(NULL == n_neurons_per_layer || 0 == n_layers) return NULL;

   //Setup MLP data
   MLP *perceptron = malloc(sizeof(MLP));
   perceptron->num_layers   = n_layers;
   perceptron->layers       = malloc(n_layers*sizeof(Layer *));

   //Create first layer manually (features are n_features inputs + 1 bias slot)
   perceptron->layers[0]    = CreateLayer(n_features + 1, n_neurons_per_layer[0]);

   //Create remaining layers (features are previous layer's outputs + 1 bias slot)
   for(int i=1; i<n_layers; i++){
      perceptron->layers[i] = CreateLayer(n_neurons_per_layer[i-1] + 1, n_neurons_per_layer[i]);
      ConnectLayers(perceptron->layers[i-1], perceptron->layers[i]);
   }

   //Connect to final layer to MLP output array
   perceptron->num_out      = perceptron->layers[n_layers-1]->num_neurons;
   perceptron->out          = perceptron->layers[n_layers-1]->out;

   return perceptron;
}

//Convenience: write per-sample input values into the shared layer-0 input Values (features[1..num_features-1]).
//count must equal the n_features passed to CreateMLP. All layer-0 neurons share the same input Value pointers via ConnectMLPInputs, so writing through neurons[0] updates them for all neurons.
void SetMLPInputs(MLP *mlp, double *xs, unsigned int count){
   if(NULL == mlp || NULL == xs || 0 == mlp->num_layers) return;
   Layer *l0 = mlp->layers[0];
   if(NULL == l0 || 0 == l0->num_neurons) return;
   Neuron *n0 = l0->neurons[0];
   if(count != n0->num_features - 1) return;
   for(unsigned int i = 0; i < count; i++) n0->features[i+1]->data = xs[i];
}

//Loads weights from a plain-text file (one float per line) in layer-major, neuron-major, weight-index order. Returns 1 on success, 0 on any failure (file open, short read, malformed value).
//Order written/read: for each layer, for each neuron, weights[0] (bias) first, then weights[1..num_features-1]. This matches the canonical iteration in CreateNeuron and lets nn_test.c and nn_test.py share an init_weights.txt for cross-implementation reproducibility.
int LoadMLPWeights(MLP *mlp, const char *filename){
   if(NULL == mlp || NULL == filename) return 0;
   FILE *f = fopen(filename, "r");
   if(NULL == f) return 0;

   for(unsigned int li = 0; li < mlp->num_layers; li++){
      Layer *l = mlp->layers[li];
      for(unsigned int ni = 0; ni < l->num_neurons; ni++){
         Neuron *n = l->neurons[ni];
         for(unsigned int wi = 0; wi < n->num_features; wi++){
            double v;
            if(fscanf(f, "%lf", &v) != 1){ fclose(f); return 0; }
            n->weights[wi]->data = v;
         }
      }
   }

   fclose(f);
   return 1;
}

//Returns a malloc'd array of every learnable Value in the MLP (every weight, with bias packed in as weights[0] of each neuron).
//Caller frees the returned array; the underlying Values remain owned by the MLP. *out_count receives the total parameter count.
Value **MLPParameters(MLP *mlp, unsigned int *out_count){
   if(NULL == mlp){ if(out_count) *out_count = 0; return NULL; }

   unsigned int total = 0;
   for(unsigned int li = 0; li < mlp->num_layers; li++){
      Layer *l = mlp->layers[li];
      for(unsigned int ni = 0; ni < l->num_neurons; ni++)
         total += l->neurons[ni]->num_features;
   }

   Value **params = malloc(total * sizeof(Value *));
   unsigned int idx = 0;
   for(unsigned int li = 0; li < mlp->num_layers; li++){
      Layer *l = mlp->layers[li];
      for(unsigned int ni = 0; ni < l->num_neurons; ni++){
         Neuron *n = l->neurons[ni];
         for(unsigned int wi = 0; wi < n->num_features; wi++)
            params[idx++] = n->weights[wi];
      }
   }

   if(out_count) *out_count = total;
   return params;
}

//Wires user-provided input Values into layer 0's neurons (features[1..]); count must equal the n_features passed to CreateMLP.
//Call this BEFORE CreateTrainer if a Trainer with cached topo will be used; afterwards, update inputs[k]->data only — do not rewire.
void ConnectMLPInputs(MLP *mlp, Value **inputs, unsigned int count){
   if(NULL == mlp || NULL == inputs || 0 == mlp->num_layers) return;

   Layer *l0 = mlp->layers[0];
   for(unsigned int j = 0; j < l0->num_neurons; j++)
      ConnectNeuronFeatures(l0->neurons[j], inputs, count);

   return;
}

void FreeMLP(MLP *mlp){
   if(NULL == mlp) return;

   for(unsigned int i = 0; i < mlp->num_layers; i++)
      FreeLayer(mlp->layers[i]);

   free(mlp->layers);
   //mlp->out aliases layers[n_layers-1]->out which FreeLayer already freed; do NOT free it again
   free(mlp);

   return;
}

//Combined squared-error loss: loss_out = sum_k (model->out[k] - targets[k])^2, a single scalar Value*.
//Targets are created with .data = 0 and meant to be overwritten per sample by the training loop.
//Note: SubValues internally creates orphan -1.0 leaves and PowValues uses orphan 2.0 leaves; both end up in trainer->topo and must be cleaned up by FreeTrainer (set-difference vs MLP's reachable Values).
void MSELoss(Trainer *t){
   if(NULL == t || NULL == t->model || NULL == t->model->out) return;

   Value *acc = NULL;
   for(unsigned int k = 0; k < t->num_out; k++){
      t->targets[k]   = CreateValue(0.0);
      Value *diff     = SubValues(t->model->out[k], t->targets[k]);
      Value *exponent = CreateValue(2.0);
      Value *sq       = PowValues(diff, exponent);

      acc = (NULL == acc) ? sq : AddValues(acc, sq);
   }
   t->loss_out = acc;

   return;
}

//Builds a Trainer over an already-wired MLP. The MLP must have its inputs hooked up via ConnectMLPInputs already,
//since this function caches a topological sort and any later re-pointering would invalidate the cache.
//loss_func is invoked once with the partially-initialised Trainer and is responsible for filling targets[k] for each output and setting loss_out to the single combined loss root.
Trainer *CreateTrainer(MLP *model, void (*loss_func)(Trainer *t)){
   if(NULL == model || NULL == loss_func || 0 == model->num_out) return NULL;

   Trainer *trainer       = malloc(sizeof(Trainer));
   trainer->model         = model;
   trainer->num_out       = model->num_out;
   trainer->loss_out      = NULL;
   trainer->targets       = malloc(model->num_out * sizeof(Value *));
   trainer->topo          = malloc(sizeof(ValueList));
   *trainer->topo         = (ValueList){0};
   trainer->learning_rate = 0.05;

   //Caller-supplied loss builder fills trainer->targets[k] and sets trainer->loss_out (single combined root)
   loss_func(trainer);

   //Cache topo from the single loss root. ugrad's stock Forward/Backward will rebuild their own topo each call;
   //this cache is here for FreeTrainer (set-difference vs MLP topo) and any future custom walkers that want to skip the rebuild.
   build_topo(trainer->loss_out, trainer->topo);

   return trainer;
}

//Convenience: write per-sample target values into trainer->targets[k]->data
void SetTrainerTargets(Trainer *t, double *ys, unsigned int count){
   if(NULL == t || NULL == ys || count != t->num_out) return;
   for(unsigned int k = 0; k < count; k++) t->targets[k]->data = ys[k];
}

//Vanilla SGD step: w->data -= learning_rate * w->grad for every weight in the MLP. Walks neurons directly to avoid an extra malloc per step.
void MLPStep(Trainer *t){
   if(NULL == t || NULL == t->model) return;
   double lr = t->learning_rate;
   MLP *mlp  = t->model;
   for(unsigned int li = 0; li < mlp->num_layers; li++){
      Layer *l = mlp->layers[li];
      for(unsigned int ni = 0; ni < l->num_neurons; ni++){
         Neuron *n = l->neurons[ni];
         for(unsigned int wi = 0; wi < n->num_features; wi++)
            n->weights[wi]->data -= lr * n->weights[wi]->grad;
      }
   }
}

//Frees the loss-side Value graph plus the Trainer's own bookkeeping. Does NOT free the underlying MLP — call FreeMLP separately afterward.
//Must be called BEFORE FreeMLP, since identifying loss-only Values requires walking model->out subgraphs (which FreeMLP would invalidate).
//Loss-only Values = trainer->topo minus everything reachable from model->out: target leaves, exponent 2.0 leaves, -1.0 leaves from NegValue, Sub-Adds, Neg-Muls, Pow heads, and the sum-chain Adds across outputs.
void FreeTrainer(Trainer *t){
   if(NULL == t) return;

   //Build the MLP-reachable set. Includes user-supplied input Values borrowed by layer-0 neurons; those are not trainer-owned and FreeMLP also leaves them alone, so excluding them from the free set is correct (caller frees their own inputs).
   ValueList mlp_topo = {0};
   if(NULL != t->model && NULL != t->model->out)
      for(unsigned int k = 0; k < t->model->num_out; k++)
         build_topo(t->model->out[k], &mlp_topo);

   //Free anything in trainer->topo that isn't MLP-reachable
   if(NULL != t->topo)
      for(size_t i = 0; i < t->topo->len; i++)
         if(!list_contains(&mlp_topo, t->topo->data[i]))
            FreeValue(t->topo->data[i]);

   free(mlp_topo.data);

   free(t->targets);
   if(NULL != t->topo){
      free(t->topo->data);
      free(t->topo);
   }
   free(t);

   return;
}

//Build the whole training stack in the correct order: CreateMLP -> allocate input Values -> ConnectMLPInputs -> CreateTrainer.
//Returns the Trainer; access the rest via t->model and t->model->layers[0]->neurons[0]->features[1..n_features] (those are the input Values).
//Caller writes per-sample inputs via features[i+1]->data and per-sample targets via t->targets[k]->data.
Trainer *CreateTrainerSystem(unsigned int n_features,
                             unsigned int n_neurons_per_layer[],
                             unsigned int n_layers,
                             void (*loss_func)(Trainer *t)){
   if(NULL == n_neurons_per_layer || 0 == n_layers || 0 == n_features || NULL == loss_func) return NULL;

   //1. Build the MLP
   MLP *model = CreateMLP(n_features, n_neurons_per_layer, n_layers);
   if(NULL == model) return NULL;

   //2. Allocate input Values; ConnectMLPInputs copies these pointers into every layer-0 neuron's features[1..], so the array itself is just a temporary buffer.
   Value **inputs = malloc(n_features * sizeof(Value *));
   for(unsigned int i = 0; i < n_features; i++) inputs[i] = CreateValue(0.0);

   //3. Wire the inputs BEFORE building the Trainer so the cached topo traces through them.
   ConnectMLPInputs(model, inputs, n_features);
   free(inputs); //Values now owned via layer-0 features (owns_features=0); the temp pointer array isn't needed

   //4. Build the Trainer; loss_func fills targets[k] and loss_out, then CreateTrainer caches the topo.
   return CreateTrainer(model, loss_func);
}

//Tear down the whole stack in the correct order: snapshot input pointers from layer 0, FreeTrainer, FreeMLP, then free inputs.
//Snapshot must happen before FreeMLP because FreeMLP destroys layer 0; FreeTrainer must run before FreeMLP because its set-difference walks model->out.
void FreeTrainerSystem(Trainer *t){
   if(NULL == t) return;

   MLP *model = t->model;

   //Recover the input Values from layer 0 (all layer-0 neurons share the same input pointers in features[1..num_features-1])
   unsigned int n_inputs    = 0;
   Value **saved_inputs     = NULL;
   if(NULL != model && model->num_layers > 0 && NULL != model->layers[0] && model->layers[0]->num_neurons > 0){
      Neuron *n0   = model->layers[0]->neurons[0];
      n_inputs     = n0->num_features - 1;
      saved_inputs = malloc(n_inputs * sizeof(Value *));
      for(unsigned int i = 0; i < n_inputs; i++) saved_inputs[i] = n0->features[i+1];
   }

   FreeTrainer(t);                     //must run before FreeMLP (set-difference walks model->out)
   if(NULL != model) FreeMLP(model);   //skips layer-0 features[1..] because owns_features=0 — inputs survive

   for(unsigned int i = 0; i < n_inputs; i++) FreeValue(saved_inputs[i]);
   free(saved_inputs);

   return;
}

#endif