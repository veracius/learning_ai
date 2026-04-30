//Karpathy-style end-to-end smoke test of the MLP training stack.
//Trains MLP(3, [4, 4, 1]) on his 4-sample tiny dataset for 20 epochs and prints loss + final predictions.
//Compile: gcc nn_test.c -lm -o nn_test
#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

int main(void){
   srand(1); //deterministic init for reproducible output

   double xs[4][3] = {
      { 2.0,  3.0, -1.0},
      { 3.0, -1.0,  0.5},
      { 0.5,  1.0,  1.0},
      { 1.0,  1.0, -1.0}
   };
   double ys[4] = {1.0, -1.0, -1.0, 1.0};

   unsigned int layer_sizes[3] = {4, 4, 1};
   Trainer *t = CreateTrainerSystem(3, layer_sizes, 3, MSELoss);
   if(NULL == t){ fprintf(stderr, "CreateTrainerSystem failed\n"); return 1; }
   if(!LoadMLPWeights(t->model, "init_weights.txt")){
      fprintf(stderr, "LoadMLPWeights failed (expected init_weights.txt next to the binary)\n");
      FreeTrainerSystem(t);
      return 1;
   }
   t->learning_rate = 0.05;

   for(int epoch = 0; epoch < 20; epoch++){
      double total_loss = 0.0;
      for(int s = 0; s < 4; s++){
         SetMLPInputs(t->model, xs[s], 3);
         SetTrainerTargets(t, &ys[s], 1);

         ZeroGrad(t->loss_out, t->topo);
         Forward(t->loss_out, t->topo);
         Backward(t->loss_out, t->topo);

         total_loss += t->loss_out->data;

         MLPStep(t);
      }
      printf("epoch %2d  loss=%g\n", epoch, total_loss);
   }

   printf("\nFinal predictions:\n");
   for(int s = 0; s < 4; s++){
      SetMLPInputs(t->model, xs[s], 3);
      Forward(t->loss_out, t->topo);
      printf("  x=(% .1f,% .1f,% .1f)  pred=% .4f  target=% .1f\n",
             xs[s][0], xs[s][1], xs[s][2],
             t->model->out[0]->data, ys[s]);
   }

   FreeTrainerSystem(t);
   return 0;
}
