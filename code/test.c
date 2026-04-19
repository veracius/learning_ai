/*
 * Reproduces the core examples from karpathy_1.ipynb using ugrad.h +
 * expression_parser.h.
 *
 * Build:  gcc -Wall -O2 -o test test.c -lm
 * Run:    ./test
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "ugrad.h"

//-----------------------------------------------------------------------------
// Small helpers for the test harness
//-----------------------------------------------------------------------------
static Value *named(Value *v, const char *label){
   strncpy(v->label, label, sizeof(v->label) - 1);
   v->label[sizeof(v->label) - 1] = '\0';
   return v;
}

// Non-deduped recursive search by label. Fine for small test graphs; returns
// the first match encountered.
static Value *find_by_label(Value *v, const char *name){
   if(NULL == v) return NULL;
   if(0 == strcmp(v->label, name)) return v;
   for(unsigned char i = 0; i < v->num_prev; i++){
      Value *hit = find_by_label(v->prev[i], name);
      if(hit) return hit;
   }
   return NULL;
}

static void section(const char *title){
   printf("\n========================================\n");
   printf("  %s\n", title);
   printf("========================================\n");
}

static void check(const char *what, double got, double want){
   double err = fabs(got - want);
   printf("  %-24s got=%-10g want=%-10g %s\n",
          what, got, want, (err < 1e-6) ? "OK" : "!! MISMATCH");
}

//-----------------------------------------------------------------------------
// 1. First notebook cell:  f(x) = 3x^2 - 4x + 5,  f(3) = 20
//    Built via the parser, so x appears twice and must share.
//-----------------------------------------------------------------------------
static void test_quadratic(void){
   section("f(x) = 3*x^2 - 4*x + 5,  x = 3  ->  20");

   const char *names[]  = {"x"};
   double      values[] = {3.0};

   Value *y = GenerateExpressionTree("3*x^2 - 4*x + 5", names, values, 1);
   if(!y){ printf("  parse failed\n"); return; }

   check("f(3)", y->data, 20.0);

   Backward(y);

   // dy/dx = 6x - 4 = 14 at x=3.  Succeeds only if both 'x' refs share one node.
   Value *x = find_by_label(y, "x");
   check("dy/dx at x=3", x->grad, 14.0);
}

//-----------------------------------------------------------------------------
// 2. Manual graph:  L = (a*b + c) * f  with a=2, b=-4, c=10, f=-2  ->  L=-4
//    Matches the cell where Karpathy builds and hand-computes gradients.
//-----------------------------------------------------------------------------
static void test_manual_L(void){
   section("Manual graph: L = (a*b + c) * f");

   Value *a = named(CreateValue( 2.0), "a");
   Value *b = named(CreateValue(-4.0), "b");
   Value *c = named(CreateValue(10.0), "c");
   Value *f = named(CreateValue(-2.0), "f");

   Value *e = named(MulValues(a, b), "e");
   Value *d = named(AddValues(e, c), "d");
   Value *L = named(MulValues(d, f), "L");

   check("L.data", L->data, -4.0);

   Backward(L);

   // Hand-derived in the notebook:
   check("a.grad",  a->grad,  8.0);
   check("b.grad",  b->grad, -4.0);
   check("c.grad",  c->grad, -2.0);
   check("d.grad",  d->grad, -2.0);
   check("e.grad",  e->grad, -2.0);
   check("f.grad",  f->grad,  2.0);

   printf("\n  Tree:\n");
   PrintTree(L);
}

//-----------------------------------------------------------------------------
// 3. One-step "optimization" from the notebook (actually gradient ASCENT:
//    data += 0.01 * grad).  Expected new L ~ -3.122064.
//-----------------------------------------------------------------------------
static void test_one_step_step(void){
   section("One ascent step: data += 0.01 * grad, then re-forward");

   Value *a = named(CreateValue( 2.0), "a");
   Value *b = named(CreateValue(-4.0), "b");
   Value *c = named(CreateValue(10.0), "c");
   Value *f = named(CreateValue(-2.0), "f");
   Value *L = MulValues(AddValues(MulValues(a, b), c), f);

   Backward(L);

   a->data += 0.01 * a->grad;
   b->data += 0.01 * b->grad;
   c->data += 0.01 * c->grad;
   f->data += 0.01 * f->grad;

   // Rebuild the graph with updated leaves
   Value *L2 = MulValues(AddValues(MulValues(a, b), c), f);
   check("L after step", L2->data, -3.122064);
}

//-----------------------------------------------------------------------------
// 4. 2D neuron:  o = tanh(x1*w1 + x2*w2 + b)
//    Gradients match the PyTorch reference in the notebook:
//      x1.grad = -1.5, w1.grad = 1.0, x2.grad = 0.5, w2.grad = 0.0
//-----------------------------------------------------------------------------
static void test_neuron(void){
   section("Neuron: o = tanh(x1*w1 + x2*w2 + b)");

   Value *x1 = named(CreateValue(2.0),          "x1");
   Value *x2 = named(CreateValue(0.0),          "x2");
   Value *w1 = named(CreateValue(-3.0),         "w1");
   Value *w2 = named(CreateValue(1.0),          "w2");
   Value *b  = named(CreateValue(6.881373587),  "b");

   Value *x1w1 = named(MulValues(x1, w1),       "x1w1");
   Value *x2w2 = named(MulValues(x2, w2),       "x2w2");
   Value *sum  = named(AddValues(x1w1, x2w2),   "x1w1+x2w2");
   Value *n    = named(AddValues(sum, b),       "n");
   Value *o    = named(TanhValue(n),            "o");

   check("o.data", o->data, 0.7071067811865);

   Backward(o);

   check("x1.grad", x1->grad, -1.5);
   check("w1.grad", w1->grad,  1.0);
   check("x2.grad", x2->grad,  0.5);
   check("w2.grad", w2->grad,  0.0);
}

//-----------------------------------------------------------------------------
// 5. Same L graph, but built via the parser.  Demonstrates that the parser
//    + symbol table give the same forward value and same leaf gradients.
//-----------------------------------------------------------------------------
static void test_parser_matches_manual(void){
   section("Parser: (a*b + c) * f  -- matches manual graph");

   const char *names[]  = {"a", "b", "c", "f"};
   double      values[] = { 2.0, -4.0, 10.0, -2.0};

   Value *L = GenerateExpressionTree("(a*b + c) * f", names, values, 4);
   if(!L){ printf("  parse failed\n"); return; }

   check("L.data", L->data, -4.0);

   Backward(L);

   check("a.grad", find_by_label(L, "a")->grad,  8.0);
   check("b.grad", find_by_label(L, "b")->grad, -4.0);
   check("c.grad", find_by_label(L, "c")->grad, -2.0);
   check("f.grad", find_by_label(L, "f")->grad,  2.0);

   printf("\n  Tree (note: SubValues/DivValues expand to intermediate nodes):\n");
   PrintTree(L);
}

//-----------------------------------------------------------------------------
// 6. The "a + a" sharing bug Karpathy flagged:  d(2a)/da = 2, not 1.
//    Our parser caches 'a' so both AddValues inputs are the same pointer;
//    AddBackward's += accumulates correctly.
//-----------------------------------------------------------------------------
static void test_shared_variable(void){
   section("Shared variable: y = a + a  ->  dy/da = 2");

   const char *names[]  = {"a"};
   double      values[] = {3.0};

   Value *y = GenerateExpressionTree("a + a", names, values, 1);
   if(!y){ printf("  parse failed\n"); return; }

   check("y.data", y->data, 6.0);

   Backward(y);
   check("dy/da",  find_by_label(y, "a")->grad, 2.0);
}

int main(void){
   test_quadratic();
   test_manual_L();
   test_one_step_step();
   test_neuron();
   test_parser_matches_manual();
   test_shared_variable();
   printf("\n");
   return 0;
}
