#ifndef EXPRESSION_PARSER_H
#define EXPRESSION_PARSER_H

/*
 * Depends on Value and the forward ops (CreateValue, AddValues, SubValues,
 * MulValues, PowValues, NegValue) from ugrad.h. Designed to be included
 * by ugrad.h AFTER those definitions; do not include directly.
 *
 * Grammar (Python-like precedence):
 *   expr   := term   (('+'|'-') term)*
 *   term   := unary  (('*'|'/') unary)*
 *   unary  := '-' unary | factor
 *   factor := primary ('^' unary)?          // right-assoc, exp accepts unary
 *   primary:= number | ident | '(' expr ')'
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXPR_MAX_SYMBOLS   64
#define EXPR_MAX_IDENT_LEN 64

Value *GenerateExpressionTree(const char *expr,
                              const char *names[],
                              double values[],
                              int n);

//-----------------------------------------------------------------------------
// Internal state
//-----------------------------------------------------------------------------
typedef enum {
   EPT_NUMBER, EPT_IDENT,
   EPT_PLUS, EPT_MINUS, EPT_STAR, EPT_SLASH, EPT_CARET,
   EPT_LPAREN, EPT_RPAREN,
   EPT_END, EPT_ERROR
} ExprTok;

typedef struct {
   ExprTok type;
   double  num_val;
   char    ident[EXPR_MAX_IDENT_LEN];
} ExprToken;

typedef struct {
   const char *src;
   size_t      pos;
   ExprToken   tok;

   const char **sym_names;
   double      *sym_values;
   int          sym_count;

   // Caches leaf Value* per identifier so repeated idents share the same node.
   char   cached_names[EXPR_MAX_SYMBOLS][EXPR_MAX_IDENT_LEN];
   Value *cached_nodes[EXPR_MAX_SYMBOLS];
   int    cached_count;

   int error;
} ExprParser;

//-----------------------------------------------------------------------------
// Lexer
//-----------------------------------------------------------------------------
static void expr_lex_next(ExprParser *p){
   while(p->src[p->pos] && isspace((unsigned char)p->src[p->pos])) p->pos++;

   char c = p->src[p->pos];
   if(c == '\0'){ p->tok.type = EPT_END; return; }

   if(isdigit((unsigned char)c) || c == '.'){
      char *end;
      p->tok.num_val = strtod(p->src + p->pos, &end);
      p->pos = (size_t)(end - p->src);
      p->tok.type = EPT_NUMBER;
      return;
   }

   if(isalpha((unsigned char)c) || c == '_'){
      size_t start = p->pos;
      while(isalnum((unsigned char)p->src[p->pos]) || p->src[p->pos] == '_') p->pos++;
      size_t len = p->pos - start;
      if(len >= EXPR_MAX_IDENT_LEN) len = EXPR_MAX_IDENT_LEN - 1;
      memcpy(p->tok.ident, p->src + start, len);
      p->tok.ident[len] = '\0';
      p->tok.type = EPT_IDENT;
      return;
   }

   p->pos++;
   switch(c){
      case '+': p->tok.type = EPT_PLUS;   return;
      case '-': p->tok.type = EPT_MINUS;  return;
      case '*': p->tok.type = EPT_STAR;   return;
      case '/': p->tok.type = EPT_SLASH;  return;
      case '^': p->tok.type = EPT_CARET;  return;
      case '(': p->tok.type = EPT_LPAREN; return;
      case ')': p->tok.type = EPT_RPAREN; return;
   }

   p->tok.type = EPT_ERROR;
   p->error    = 1;
}

//-----------------------------------------------------------------------------
// Symbol lookup + leaf caching (gives DAG sharing of repeated idents)
//-----------------------------------------------------------------------------
static double expr_symbol_value(ExprParser *p, const char *name){
   for(int i = 0; i < p->sym_count; i++)
      if(strcmp(p->sym_names[i], name) == 0) return p->sym_values[i];
   return 0.0;
}

static Value *expr_get_or_create_leaf(ExprParser *p, const char *name){
   for(int i = 0; i < p->cached_count; i++)
      if(strcmp(p->cached_names[i], name) == 0) return p->cached_nodes[i];

   Value *v = CreateValue(expr_symbol_value(p, name));
   strncpy(v->label, name, sizeof(v->label) - 1);
   v->label[sizeof(v->label) - 1] = '\0';

   if(p->cached_count < EXPR_MAX_SYMBOLS){
      strncpy(p->cached_names[p->cached_count], name, EXPR_MAX_IDENT_LEN - 1);
      p->cached_names[p->cached_count][EXPR_MAX_IDENT_LEN - 1] = '\0';
      p->cached_nodes[p->cached_count] = v;
      p->cached_count++;
   }
   return v;
}

//-----------------------------------------------------------------------------
// Recursive-descent parser
//-----------------------------------------------------------------------------
static Value *expr_parse_expr(ExprParser *p);

static Value *expr_parse_primary(ExprParser *p){
   if(p->tok.type == EPT_NUMBER){
      Value *v = CreateValue(p->tok.num_val);
      expr_lex_next(p);
      return v;
   }
   if(p->tok.type == EPT_IDENT){
      Value *v = expr_get_or_create_leaf(p, p->tok.ident);
      expr_lex_next(p);
      return v;
   }
   if(p->tok.type == EPT_LPAREN){
      expr_lex_next(p);
      Value *v = expr_parse_expr(p);
      if(!v) return NULL;
      if(p->tok.type != EPT_RPAREN){ p->error = 1; return NULL; }
      expr_lex_next(p);
      return v;
   }
   p->error = 1;
   return NULL;
}

static Value *expr_parse_unary(ExprParser *p);

static Value *expr_parse_factor(ExprParser *p){
   Value *base = expr_parse_primary(p);
   if(!base) return NULL;
   if(p->tok.type == EPT_CARET){
      expr_lex_next(p);
      Value *exp = expr_parse_unary(p);
      if(!exp) return NULL;
      return PowValues(base, exp);
   }
   return base;
}

static Value *expr_parse_unary(ExprParser *p){
   if(p->tok.type == EPT_MINUS){
      expr_lex_next(p);
      Value *operand = expr_parse_unary(p);
      if(!operand) return NULL;
      return NegValue(operand);
   }
   return expr_parse_factor(p);
}

static Value *expr_parse_term(ExprParser *p){
   Value *left = expr_parse_unary(p);
   if(!left) return NULL;
   while(p->tok.type == EPT_STAR || p->tok.type == EPT_SLASH){
      ExprTok op = p->tok.type;
      expr_lex_next(p);
      Value *right = expr_parse_unary(p);
      if(!right) return NULL;
      if(op == EPT_STAR){
         left = MulValues(left, right);
      } else {
         Value *neg_one = CreateValue(-1.0);
         Value *inv     = PowValues(right, neg_one);
         left           = MulValues(left, inv);
      }
   }
   return left;
}

static Value *expr_parse_expr(ExprParser *p){
   Value *left = expr_parse_term(p);
   if(!left) return NULL;
   while(p->tok.type == EPT_PLUS || p->tok.type == EPT_MINUS){
      ExprTok op = p->tok.type;
      expr_lex_next(p);
      Value *right = expr_parse_term(p);
      if(!right) return NULL;
      left = (op == EPT_PLUS) ? AddValues(left, right) : SubValues(left, right);
   }
   return left;
}

//-----------------------------------------------------------------------------
// Public entry
//-----------------------------------------------------------------------------
Value *GenerateExpressionTree(const char *expr,
                              const char *names[],
                              double values[],
                              int n){
   if(NULL == expr) return NULL;

   ExprParser p = {0};
   p.src          = expr;
   p.pos          = 0;
   p.sym_names    = names;
   p.sym_values   = values;
   p.sym_count    = n;
   p.cached_count = 0;
   p.error        = 0;

   expr_lex_next(&p);
   Value *root = expr_parse_expr(&p);

   if(p.error || p.tok.type != EPT_END){
      fprintf(stderr, "expression parse error near position %zu in: %s\n",
              p.pos, expr);
      return NULL;
   }
   return root;
}

#endif
