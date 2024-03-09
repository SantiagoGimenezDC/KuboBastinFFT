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

  r_type sysLength_;
  r_type sysSubLength_;

public:
  ~Device(){};
  Device(device_vars& device_vars):device_vars_(device_vars),rng_(device_vars.dis_seed_){};


  void set_sysLength(r_type sysLength){sysLength_=sysLength;};
  r_type sysLength(){return sysLength_;};
  void set_sysSubLength(r_type sysSubLength){sysSubLength_=sysSubLength;};
  r_type sysSubLength(){return sysSubLength_;};
  virtual r_type Hamiltonian_size() = 0;  

  virtual void build_Hamiltonian() = 0;
  virtual void damp   ( r_type*) = 0;
  virtual void update_dis(r_type*,r_type*) = 0; 
  virtual void adimensionalize ( r_type,  r_type ) = 0;
  
  virtual void rearrange_initial_vec(type*) = 0; //very hacky
  virtual void traceover(type*, type*, int, int) = 0;
  
  Random& rng(){return rng_;};
  //  CAP& cap(){return cap_;};
  device_vars& parameters(){return device_vars_; };


  virtual void update_cheb_filtered ( type*, type*, type*, r_type*, r_type* , type) {};  

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
