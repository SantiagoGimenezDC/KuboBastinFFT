#ifndef GRAPHENE_HPP
#define GRAPHENE_HPP

#include "../static_vars.hpp"
#include "../CAP.hpp"
#include "../Random.hpp"
#include "Device.hpp"

class Graphene: public Device{
private:
  const r_type t_standard_ = -2.7;
  bool CYCLIC_BCs_ = false;

  r_type a_ = 1.0,
    b_ = 0.0,
    t_a_= t_standard_;
  
public:
  ~Graphene(){};
  Graphene(device_vars&);

  virtual void adimensionalize(r_type a, r_type b){ a_ = a, b_ = b; t_a_/=a;};
  virtual void build_Hamiltonian(){};
  virtual void damp( r_type*){};
  

  virtual void traceover(type*, type*, int, int);


  
  

  virtual  void update_cheb ( type*, type*, type*, r_type*, r_type*);  

  virtual void update_cheb ( type*, type*, type*, r_type*, r_type , r_type ){};
  virtual void update_cheb ( int ,  int, type*, type*, type*, type*, r_type*, r_type , r_type ){};

  virtual void H_ket ( type*, type*);
  virtual void H_ket ( type*, type*, r_type*, r_type*);  

  virtual void vel_op (type*, type*);
  virtual void setup_velOp(){};
  
  void vertical_BC(type*, type*, r_type*);
  void horizontal_BC(type*, type*, r_type* );

};





#endif //GRAPHENE_H
