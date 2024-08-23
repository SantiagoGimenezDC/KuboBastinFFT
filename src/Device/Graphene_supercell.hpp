#ifndef GRAPHENE_SUPERCELL_HPP
#define GRAPHENE_SUPERCELL_HPP

#include "../static_vars.hpp"
#include "../CAP.hpp"
#include "../Random.hpp"
#include "Device.hpp"
#include "Graphene.hpp"
#include <eigen-3.4.0/Eigen/Sparse>
#include "Coordinates.hpp"
#include <iostream>

class Graphene_supercell: public Graphene{

private:
  const r_type t_standard_ = -2.7;
  bool CYCLIC_BCs_ = false;
  int fullLe_=0;

  r_type peierls_d_=0;
  
  r_type t_a_= t_standard_;

public:
  ~Graphene_supercell(){};
  //  Graphene_supercell(device_vars&);

  Graphene_supercell(device_vars& ) ;

  void print_hamiltonian();

  
  //Generic interfaces

  virtual void update_cheb ( type* , type* , type* );
  
  virtual void H_ket (  type*, type* );



  virtual void vel_op   ( type* ket, type* p_ket, int dir){
    if( dir == 0 )
      vel_op_x( ket, p_ket);
    if( dir == 1 )
      vel_op_y( ket, p_ket);
  };
  virtual void vel_op_y (type* , type* );
  virtual void vel_op_x (type* , type* );
  
  type peierls(int i1, int sign){
    return  std::polar(1.0,  ( i1 % 2 == 0 ? -1 : 1 ) * peierls_d_ * ( 2 * i1 + sign * 1  ) );
  };

  //----On the fly implementations
  
  
  void vertical_BC(r_type, type*, type*, r_type*);
  void horizontal_BC(r_type, type*, type*, r_type* );


  


  



  //----Sparse matrix representation

  inline r_type y(int i, int ){  return -i*cos(M_PI/6.0); };

  
  inline r_type x(int i, int j){
    r_type x_p=0;

    if( i % 2 == 1 )
      x_p = 1 * ( sin(M_PI/6.0)  +  (j/2)*(1.0+2.0*sin(M_PI/6.0))  +  ((j+1)/2)   );	//The 1 * should be an a0_;                          	                            
    else
      x_p = 1 * (  ((j+1)/2)*(1.0+2.0*sin(M_PI/6.0)) + (j/2)  );	                       

    return x_p;
  };
};





#endif //GRAPHENE_SUPERCELL_H
