#ifndef GRAPHENE_HPP
#define GRAPHENE_HPP

#include "static_vars.hpp"
#include "CAP.hpp"

struct device_vars{
  int W_, LE_, C_, DIM_, SUBDIM_;
};

class Graphene{
private:
  device_vars graphene_vars_;
  bool CYCLIC_BCs_ = false;
public:
  ~Graphene(){};
  Graphene(){};
  Graphene(device_vars&);

  //  CAP& cap(){return cap_;};
  device_vars& parameters(){return graphene_vars_; };
  void update_cheb ( type*, type*, type*, type*, type , type );
  void update_cheb ( int ,  int, type*, type*, type*, type*, type*, type , type );
  void vel_op (type*, type*);

  void vertical_BC(type*, type*, type*, type );
  void horizontal_BC(type*, type*, type*, type );

};





#endif //GRAPHENE_H
