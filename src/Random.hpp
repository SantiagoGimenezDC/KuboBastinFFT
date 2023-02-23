#ifndef RANDOM_HPP
#define RANDOM_HPP


#include<random>
#include "static_vars.hpp"


class Random{
private:
  int seed_;
  std::mt19937 rng_;
  std::uniform_real_distribution <r_type> dist_;

public:
  virtual ~Random(){};
  Random(int seed) : seed_(seed){
      rng_.seed(seed_);
  };

  r_type get(){return dist_(rng_);};


};




#endif //RAND_BASE_HPP
