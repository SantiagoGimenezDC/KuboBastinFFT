#ifndef RAND_BASE_HPP
#define RAND_BASE_HPP

#include<random>
#include "static_vars.hpp"
#include "Device/Device.hpp"
#include "Random.hpp"

void generate_vec(int, int,int, type*, int , int );
void generate_vec_im(int, int,int, type*, int , int );


class Vec_Base{
private:
  device_vars& parameters_;
  Random rng_;
public:
  virtual ~Vec_Base(){};
  Vec_Base(device_vars& parameters, int seed) : parameters_(parameters), rng_(seed){};

  Random& rng(){return rng_;};
  device_vars& parameters(){return parameters_;};
  
  virtual void generate_vec_re( r_type*, int ) = 0;
  virtual void generate_vec_im(std::complex<r_type>*, int ) = 0;  
};




class Complex_Phase: public Vec_Base{
public:
  virtual ~Complex_Phase(){};
  Complex_Phase(device_vars& parameters, int seed) : Vec_Base(parameters, seed){};
  virtual void generate_vec_re ( r_type*, int );
  virtual void generate_vec_im ( std::complex<r_type>*, int );  
};



class Complex_Phase_real: public Vec_Base{
public:
  virtual ~Complex_Phase_real(){};
  Complex_Phase_real(device_vars& parameters, int seed) : Vec_Base(parameters, seed){};
  virtual void generate_vec_re ( r_type*, int );
  virtual void generate_vec_im ( std::complex<r_type>*, int );  
};



class Direct: public Vec_Base{
public:
  virtual ~Direct(){};
  Direct(device_vars& parameters, int seed) : Vec_Base(parameters, seed){};
  
  virtual void generate_vec_re(r_type*, int );
  virtual void generate_vec_im( std::complex<r_type>*, int );  
};


class FullTrace: public Vec_Base{
public:
  virtual ~FullTrace(){};
  FullTrace(device_vars& parameters, int seed) : Vec_Base(parameters, seed){};
  
  virtual void generate_vec_re(r_type*, int );
  virtual void generate_vec_im( std::complex<r_type>*, int );  
};


class projected_FullTrace: public Vec_Base{
private:
  int stride_;
public:
  virtual ~projected_FullTrace(){};
  projected_FullTrace(device_vars& parameters, int seed, int stride) : Vec_Base(parameters, seed), stride_(stride){};
  
  virtual void generate_vec_re(r_type*, int );
  virtual void generate_vec_im( std::complex<r_type>*, int );  
};


#endif //RAND_BASE_HPP
