#ifndef GRAPHENE_HPP
#define GRAPHENE_HPP

#include "../static_vars.hpp"
#include "../CAP.hpp"
#include "../Random.hpp"
#include "Device.hpp"
#include <eigen-3.4.0/Eigen/Sparse>
#include "Coordinates.hpp"
#include <iostream>

class Graphene: public Device{

  typedef Eigen::SparseMatrix<r_type,Eigen::RowMajor> SpMatrixXp;
  typedef Eigen::Matrix<r_type,-1,-1> MatrixXp;
  typedef Eigen::Matrix<type, -1, 1>                 VectorXdT;


private:
  const r_type t_standard_ = -2.7;
  bool CYCLIC_BCs_ = false;

  bool print_CSR=false;
  bool csc_mode = false;
  SpMatrixXp H_, vx_; 
  Coordinates coordinates_;

  int fullLe_;
  
  r_type a_ = 1.0,
    b_ = 0.0,
    t_a_= t_standard_;
  const r_type  a0_      = 0.142;
  const r_type  d0_     = 0.335;
  const r_type  VppPI_  = -2.7;
  const r_type  VppSIG_ = 0.48;
  const r_type  delta_  = 0.184*a0_*sqrt(3.0);

public:
  ~Graphene(){};
  Graphene(device_vars&);

  virtual r_type Hamiltonian_size(){ return ( H_.innerSize() + H_.nonZeros() + H_.outerSize() ) * sizeof(r_type); };
  
  virtual void adimensionalize(r_type a, r_type b){
    a_ = a, b_ = b; t_a_=t_standard_/a;

    if(csc_mode){

      int Dim = this->parameters().DIM_;
      SpMatrixXp Id(Dim,Dim);
      Id.setIdentity();

      H_=(H_+b*Id)/a;
    }
      
  };
  

  r_type a(){ return a_; };
  r_type b(){ return b_; };
  
  //Setting up
  virtual void damp( r_type* new_damp){
    set_damp_op(new_damp);
    if(csc_mode)
      damp_csc( new_damp);
  };
  virtual void damp_csc( r_type*);
  
  virtual void update_dis(r_type*){ Anderson_disorder(); };

  virtual void rearrange_initial_vec(type*); //very hacky
  virtual void traceover(type*, type*, int, int);


  //Generic interfaces

  virtual void update_cheb ( type* vec, type* p_vec, type* pp_vec){
    if(csc_mode)
      update_cheb_csc( vec, p_vec, pp_vec, damp_op(), dis() );
    else
      update_cheb_otf( vec, p_vec, pp_vec, damp_op(), dis() );
  };
  
  virtual void H_ket (  type* vec, type* p_vec){
    if(csc_mode)
      H_ket_csc( vec,  p_vec, damp_op(), dis() );
    else
      H_ket_otf( vec,  p_vec, damp_op(), dis() );
  };



  
  virtual void update_cheb ( type* vec, type* p_vec, type* pp_vec, r_type* dmp_op, r_type* dis_vec){
    if(csc_mode)
      update_cheb_csc( vec, p_vec, pp_vec, dmp_op, dis_vec);
    else
      update_cheb_otf( vec, p_vec, pp_vec, dmp_op, dis_vec);
  };
  
  virtual void H_ket (  type* vec, type* p_vec, r_type* dmp_op, r_type* dis_vec){
    if(csc_mode)
      H_ket_csc( vec,  p_vec, dmp_op, dis_vec);
    else
      H_ket_otf( vec,  p_vec, dmp_op, dis_vec);
  };
  
  virtual void vel_op (type* vec, type* p_vec){
    if(csc_mode)
      vel_op_otf( vec,  p_vec);
    else
      vel_op_otf( vec,  p_vec);
  };

  virtual void vel_op   ( type* ket, type* p_ket, int dir){
    if( dir == 0 )
      vel_op( ket, p_ket);
    if( dir == 1 )
      vel_op_y( ket, p_ket);
  };
  virtual void vel_op_y (type* , type* );
  
  

  //----On the fly implementations
  virtual  void update_cheb_filtered ( type*, type*, type*, r_type*, r_type*, type);


  void update_cheb_otf ( type*, type*, type*, r_type*, r_type*);
  void H_ket_otf ( type*, type*, r_type*, r_type*);
  void vel_op_otf (type*, type*);

  
  void vertical_BC(type*, type*, r_type*);
  void horizontal_BC(type*, type*, r_type* );


  

  //Unused
  //  virtual void H_ket ( type*, type*);
  virtual void update_cheb ( type*, type*, type*, r_type*, r_type , r_type ){};
  virtual void update_cheb ( int ,  int, type*, type*, type*, type*, r_type*, r_type , r_type ){};





  



  //----Sparse matrix representation
  virtual void build_Hamiltonian(){if(csc_mode) SlaterCoster_Hamiltonian();};
  virtual void setup_velOp();
  
  void single_neighbor_Hamiltonian   ();
  void SlaterCoster_Hamiltonian   ();
  void update_cheb_csc( type*, type*, type*, r_type*, r_type* );
  void H_ket_csc ( type*, type*, r_type*, r_type*);
  void vel_op_csc (type*, type*);




  virtual void set_coordinates();


  
  inline
  r_type SlaterCoster_intralayer_coefficient( r_type d_ij){
    return VppPI_  * exp( - ( d_ij - a0_) / delta_ );
  };

  inline
  r_type SlaterCoster_coefficient(Eigen::Matrix<r_type,3,1> R_ij, r_type d_ij){
    return VppPI_  * exp(-(d_ij-a0_)/delta_) * (1.0-pow(R_ij.dot(Eigen::Matrix<r_type,3,1>(0.0,0.0,1.0))/d_ij,2.0))+
      VppSIG_ * exp(-(d_ij-d0_)/delta_) *      pow(R_ij.dot(Eigen::Matrix<r_type,3,1>(0.0,0.0,1.0))/d_ij,2.0);
  };


};





#endif //GRAPHENE_H
