#ifndef DEVICE_HPP
#define DEVICE_HPP

#include "../static_vars.hpp"
#include "../CAP.hpp"
#include "../Random.hpp"

struct device_vars{
  int W_, LE_, C_, DIM_, SUBDIM_,  dis_seed_;
  r_type dis_str_, theta_, d_min_;
};


class Device{
private:
  device_vars device_vars_;
  Random rng_;

public:
  ~Device(){};
  Device(device_vars& device_vars):device_vars_(device_vars),rng_(device_vars.dis_seed_){};


  virtual void build_Hamiltonian() = 0;
  virtual void damp   ( r_type*) = 0;
  virtual void adimensionalize ( r_type,  r_type ) = 0;
  

  virtual void traceover(type*, type*, int, int) = 0;
  
  Random& rng(){return rng_;};
  //  CAP& cap(){return cap_;};
  device_vars& parameters(){return device_vars_; };

  virtual void update_cheb ( type*, type*, type*, r_type*, r_type* ) = 0;  
  virtual void update_cheb ( type*, type*, type*, r_type*, r_type , r_type ){};
  virtual void update_cheb ( int ,  int, type*, type*, type*, type*, r_type*, r_type , r_type ){};

  virtual void H_ket ( type*, type*) = 0;
  virtual void H_ket ( type*, type*, r_type*, r_type*) = 0;  

  virtual void vel_op (type*, type*)=0;
  virtual void setup_velOp() = 0;
  
  void Anderson_disorder(r_type*);  

  void minMax_EigenValues( int , r_type& , r_type& );
};


#endif //Device_HPP
