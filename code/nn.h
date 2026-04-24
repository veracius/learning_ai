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
   Value **out;
}MLP;


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
   perceptron->out          = perceptron->layers[n_layers-1]->out;

   return perceptron;
}

//Wires user-provided input Values into layer 0's neurons (features[1..]); count must equal the n_features passed to CreateMLP
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

#endif