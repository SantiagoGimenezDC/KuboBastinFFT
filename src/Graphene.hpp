#ifndef GRAPHENE_HPP
#define GRAPHENE_HPP

#include "static_vars.hpp"
#include "CAP.hpp"
#include "Random.hpp"

struct device_vars{
  int W_, LE_, C_, DIM_, SUBDIM_,  dis_seed_;
  r_type dis_str_;
};

class Graphene{
private:
  const r_type t_standard_ = 2.7;
  device_vars graphene_vars_;
  Random rng_;
  bool CYCLIC_BCs_ = false;

public:
  ~Graphene(){};
  Graphene(device_vars&);

  Random& rng(){return rng_;};
  //  CAP& cap(){return cap_;};
  device_vars& parameters(){return graphene_vars_; };
  void update_cheb ( type*, type*, type*, r_type*, r_type*, r_type , r_type );  
  void update_cheb ( type*, type*, type*, r_type*, r_type , r_type );
  void update_cheb ( int ,  int, type*, type*, type*, type*, r_type*, r_type , r_type );
  void vel_op (type*, type*);
  
  void vertical_BC(type*, type*, r_type*, r_type );
  void horizontal_BC(type*, type*, r_type*, r_type );


  void Anderson_disorder(r_type*);  
};





#endif //GRAPHENE_H
