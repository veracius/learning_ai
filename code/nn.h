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
}Neuron;

typedef struct Layer{
   unsigned int num_neurons;
   Neuron *neurons;
   
}Layer;

typedef struct MLP{

}MLP;

Neuron* CreateNeuron(unsigned int nf){
   Neuron *out = malloc(sizeof(Neuron));
   out->num_features = nf;
   out->weights = malloc(nf*sizeof(Value *));
   for(int i=0; i<nf; i++) out->weights[i] = CreateValue(((double)rand() / RAND_MAX) * 2.0 - 1.0);
   
   //Create the Value data tree for the Neuron itself
   out->features = malloc(nf*sizeof(Value *));
   out->products = malloc(nf*sizeof(Value *));
   for(int i=0; i<nf; i++){
      out->features[i] = CreateValue(1.0); //All demo features are 1.0f to start off; x_bias should stay 1.0f
      out->products[i] = MulValues(out->weights[i], out->features[i]);
   }
   Value *sum = SumValues(out->products, nf);
   out->out = TanhValue(sum);

   return out;
}

void SetNeuronWeights(Neuron *n, unsigned int count, float *weights){
   if(NULL == n || NULL == weights || count != n->num_features) return;
   for(unsigned int i = 0; i < count; i++) n->weights[i]->data = (double)weights[i];

   return;
}

void ConnectNeuronFeatures(Neuron *n, Value **v, unsigned int count){
   if(NULL == n || NULL == v || count != n->num_features) return;
   for(unsigned int i = 0; i < count; i++){
      FreeValue(n->features[i]); //orphaned after rewire; free to avoid leak
      n->features[i]          = v[i];
      n->products[i]->prev[1] = v[i];
   }
}


//Implement a Free Neuron/MLP scheme
#endif