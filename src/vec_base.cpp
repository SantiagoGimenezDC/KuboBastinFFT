#include<random>
#include "static_vars.hpp"
#include"vec_base.hpp"


void generate_vec( type rand_vec[DIM_], int seed, int r){

  std::mt19937 gen;
  gen.seed(seed+r);
  std::uniform_int_distribution<> dis(0, SUBDIM_);
  int random_site=dis(gen);

  for(int m=0; m < DIM_; m++)
    rand_vec[m] = 0;
  
  rand_vec[random_site+C_*W_] = 1.0;
  
}
