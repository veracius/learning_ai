#ifndef UGRAD_H
#define UGRAD_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Value{
   double data;
   double grad;
   unsigned char num_prev; //limit to 1 or 2 for simplicity
   struct Value *prev[2];
   void (*forward)(struct Value *curr);
   void (*backward)(struct Value *curr);
   char label[64];
} Value;

//=============================================================================
//Function Prototypes
//=============================================================================
Value *CreateValue(double data);

Value *AddValues(Value *v1, Value *v2);
void   AddForward(Value *curr);
void   AddBackward(Value *curr);

Value *MulValues(Value *v1, Value *v2);
void   MulForward(Value *curr);
void   MulBackward(Value *curr);

Value *PowValues(Value *v1, Value *v2);
void   PowForward(Value *curr);
void   PowBackward(Value *curr);

Value *TanhValue(Value *v);
void   TanhForward(Value *curr);
void   TanhBackward(Value *curr);

Value *NegValue(Value *v1);
Value *SubValues(Value *v1, Value *v2);
Value *SumValues(Value **values, size_t n);

void Forward(Value *root);
void Backward(Value *root);
void ZeroGrad(Value *root);
void FreeValue(Value *v);
void FreeGraph(Value *root);
void PrintValue(Value *v);
void PrintTree(Value *root);

//=============================================================================
//Functions
//=============================================================================
Value *CreateValue(double data){
   Value *result = malloc(sizeof(Value));
   *result = (Value){.data = data,
                     .grad = 0.0,
                     .num_prev = 0,
                     .prev = {[0]=NULL, [1]=NULL},
                     .forward = NULL,
                     .backward = NULL};
   return result;
}

Value* AddValues(Value *v1, Value *v2){
   if(NULL == v1|| NULL == v2) return NULL;
   Value *result = malloc(sizeof(Value));
   if(NULL == result) return NULL;

   *result = (Value){.data = v1->data + v2->data,
                     .grad = 0.0,
                     .num_prev = 2,
                     .prev = {[0]=v1, [1]=v2},
                     .forward = AddForward,
                     .backward = AddBackward};
   strncpy(result->label, "+", sizeof(result->label)-1);

   return result;
}

void AddForward(Value *curr){
   curr->data = curr->prev[0]->data + curr->prev[1]->data;
}

void AddBackward(Value *curr){
   
   curr->prev[0]->grad += 1.0*curr->grad;
   curr->prev[1]->grad += 1.0*curr->grad;
   return;
}

Value* MulValues(Value *v1, Value *v2){
   if(NULL == v1|| NULL == v2) return NULL;
   Value *result = malloc(sizeof(Value));
   if(NULL == result) return NULL;

   *result = (Value){.data = v1->data * v2->data,
                     .grad = 0.0,
                     .num_prev = 2,
                     .prev = {[0]=v1, [1]=v2},
                     .forward = MulForward,
                     .backward = MulBackward};
   strncpy(result->label, "*", sizeof(result->label)-1);

   return result;
}

void MulForward(Value *curr){
   curr->data = curr->prev[0]->data * curr->prev[1]->data;
}

void MulBackward(Value *curr){
   Value *v1 = curr->prev[0];
   Value *v2 = curr->prev[1];
   v1->grad += v2->data*curr->grad;
   v2->grad += v1->data*curr->grad;
}

//v1**v2 ==> pow(base, exponent)
//Need to elevate power to a Value (instead of float);
//C isn't as flexible with types as python
Value* PowValues(Value *v1, Value *v2){
   if(NULL == v1|| NULL == v2) return NULL;
   Value *result = malloc(sizeof(Value));
   if(NULL == result) return NULL;

   *result = (Value){.data = pow(v1->data, v2->data),
                     .grad = 0.0,
                     .num_prev = 2,
                     .prev = {[0]=v1, [1]=v2},
                     .forward = PowForward,
                     .backward = PowBackward};
   strncpy(result->label, "**", sizeof(result->label)-1);

   return result;
}

void PowForward(Value *curr){
   curr->data = pow(curr->prev[0]->data, curr->prev[1]->data);
}

//Elevating power to Value data object requies computation of d/dv2(func)
//d/db(a^b) = (a^b)ln(a)
void PowBackward(Value *curr){
   Value *v1 = curr->prev[0];
   Value *v2 = curr->prev[1];
   v1->grad += v2->data*pow(v1->data, (v2->data-1))*curr->grad;
   v2->grad += pow(v1->data, v2->data)*log(v1->data)*curr->grad;
}

Value* TanhValue(Value *v){
   if(NULL == v) return NULL;
   Value *result = malloc(sizeof(Value));
   if(NULL == result) return NULL;

   *result = (Value){.data = tanh(v->data),
                     .grad = 0.0,
                     .num_prev = 1,
                     .prev = {[0]=v, [1]=NULL},
                     .forward = TanhForward,
                     .backward = TanhBackward};
   strncpy(result->label, "tanh", sizeof(result->label)-1);

   return result;
}

void TanhForward(Value *curr){
   curr->data = tanh(curr->prev[0]->data);
}

void TanhBackward(Value *curr){
   Value *v = curr->prev[0];
   v->grad += (1-pow(curr->data, 2))*curr->grad;
}

//=============================================================================
//Derivative Operations
//=============================================================================
Value *NegValue(Value *v1){
   if(NULL == v1) return NULL;
   return MulValues(v1, CreateValue(-1.0));
}

Value *SubValues(Value *v1, Value *v2){
   return AddValues(v1, NegValue(v2));
}

//not the most efficient but should be fine for neuron's sum volume
Value *SumValues(Value **values, size_t n){
   if(NULL == values || 0 == n) return NULL;
   Value *acc = values[0];
   for(size_t i = 1; i < n; i++)
      acc = AddValues(acc, values[i]);
   return acc;
}

//=============================================================================
//Backprop & Helper
//=============================================================================

typedef struct {
   Value **data;
   size_t len;
   size_t cap;
} ValueList;

static int list_contains(ValueList *l, Value *v){
   for(size_t i = 0; i < l->len; i++)
      if(l->data[i] == v) return 1;
   return 0;
}

static void list_push(ValueList *l, Value *v){
   if(l->len == l->cap){
      l->cap = l->cap ? l->cap * 2 : 16;
      l->data = realloc(l->data, l->cap * sizeof(Value *));
   }
   l->data[l->len++] = v;
}

static void build_topo(Value *v, ValueList *topo){
   if(list_contains(topo, v)) return;
   for(unsigned char i = 0; i < v->num_prev; i++)
      build_topo(v->prev[i], topo);
   list_push(topo, v);
}

void Forward(Value *root){
   if(NULL == root) return;

   ValueList topo = {0};
   build_topo(root, &topo);

   for(size_t i = 0; i < topo.len; i++)
      if(topo.data[i]->forward) topo.data[i]->forward(topo.data[i]);

   free(topo.data);
}

void Backward(Value *root){
   if(NULL == root) return;

   ValueList topo = {0};
   build_topo(root, &topo);

   root->grad = 1.0;
   for(size_t i = topo.len; i-- > 0; )
      if(topo.data[i]->backward) topo.data[i]->backward(topo.data[i]);

   free(topo.data);
}

void ZeroGrad(Value *root){
   if(NULL == root) return;

   ValueList topo = {0};
   build_topo(root, &topo);

   for(size_t i = 0; i < topo.len; i++)
      topo.data[i]->grad = 0.0;

   free(topo.data);
}

void FreeValue(Value *v){
   free(v);
}

void FreeGraph(Value *root){
   if(NULL == root) return;

   ValueList topo = {0};
   build_topo(root, &topo);

   for(size_t i = 0; i < topo.len; i++)
      free(topo.data[i]);

   free(topo.data);
}

void PrintValue(Value *v){
   if(NULL == v){
      printf("Value(null)\n");
      return;
   }
   printf("Value(label=\"%s\", data=%g, grad=%g, num_prev=%u)\n",
          v->label, v->data, v->grad, v->num_prev);
}

static void print_tree_rec(Value *v, const char *prefix, int is_last){
   const char *connector = is_last ? "└── " : "├── ";
   printf("%s%s[%s] data=%g grad=%g\n",
          prefix, connector, v->label, v->data, v->grad);

   char new_prefix[512];
   snprintf(new_prefix, sizeof(new_prefix), "%s%s",
            prefix, is_last ? "    " : "│   ");

   for(unsigned char i = 0; i < v->num_prev; i++)
      print_tree_rec(v->prev[i], new_prefix, i == v->num_prev - 1);
}

void PrintTree(Value *root){
   if(NULL == root){ printf("(null)\n"); return; }
   printf("[%s] data=%g grad=%g\n", root->label, root->data, root->grad);
   for(unsigned char i = 0; i < root->num_prev; i++)
      print_tree_rec(root->prev[i], "", i == root->num_prev - 1);
}

#include "expression_parser.h"

#endif
