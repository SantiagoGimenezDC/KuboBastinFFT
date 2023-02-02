#include<random>
#include "static_vars.hpp"
#include"vec_base.hpp"


void generate_vec(int C, int W, int LE, type rand_vec[], int seed, int r){

  int DIM = (2*C+LE)*W, SUBDIM = LE*W;
  
  std::mt19937 gen;
  gen.seed(seed+r);
  std::uniform_int_distribution<> dis(0, SUBDIM);
  int random_site=dis(gen);

  for(int m=0; m < DIM; m++)
    rand_vec[m] = 0;
  
  rand_vec[random_site+C*W] = 1.0;
  
}
