#ifndef KUBO_BASTIN_SOLVER_TRADITIONAL_HPP
#define KUBO_BASTIN_SOLVER_TRADITIONAL_HPP

#include<string>
#include<iostream>
#include<eigen-3.4.0/Eigen/Core>

#include"../../static_vars.hpp"
#include "../../Device/Device.hpp"
#include "../../Device/Graphene.hpp"
#include "../../kernel.hpp"
#include "../../vec_base.hpp"

#include "../solver_vars.hpp"


class Kubo_solver_traditional{
private:
  solver_vars parameters_;
  formula sym_formula_;

  
  Device&  device_;
  Kernel*   kernel_;
  CAP*      cap_;
  Vec_Base* vec_base_;

  

  
//---------------Large vectors----------------//  
  Eigen::Matrix<type, -1,-1, Eigen::ColMajor> mu_, mu_r_, bras_, kets_ ;
  
  //Recursion Vectors; 4 of these are actually not needed;
  type *vec_,
       *p_vec_,
       *pp_vec_,
       *vec_2_,
       *p_vec_2_,
       *pp_vec_2_,
       *rand_vec_,
       *tmp_;
  
  //Auxiliary - disorder and CAP vectors
  r_type *dmp_op_,
         *dis_vec_ ;
//--------------------------------------------//


  
  
//---------------Dataset vectors--------------//
  r_type *E_points_,
         *conv_R_;
  type *r_data_,
       *final_data_ ;
//--------------------------------------------//

  
public:
  Kubo_solver_traditional();
  Kubo_solver_traditional( solver_vars&, Device&);
  ~Kubo_solver_traditional();
  
  solver_vars& parameters(){ return parameters_; };

  //Initializers
  void allocate_memory();
  void reset_bra_recursion_vectors();
  void reset_ket_recursion_vectors();
  void reset_Chebyshev_buffers();


  //Heavy duty
  void compute();
  void polynomial_cycle_bra(int,  int );
  void polynomial_cycle_ket(int,  int );  
  void update_cheb_moments(int ,int , int, int );

  
  //Post-process
  void Bastin_postProcess ();
  void Greenwood_postProcess ();  
  type green(int n, type energy);
  type dgreen(int n, type energy);
  Eigen::Matrix<type, -1,-1, Eigen::ColMajor> fill_green(r_type E_points[], int M, int E);
  Eigen::Matrix<type, -1,-1, Eigen::RowMajor> fill_dgreen(r_type E_points[], int M, int E);

  
  void integration ( r_type*, r_type*, r_type* );
  void partial_integration ( r_type*, r_type*, r_type* );

  void rearrange_crescent_order( r_type* );
  void eta_CAP_correct(r_type*, r_type* );  
  void plot_data   ( std::string, std::string );




};


#endif //KUBO_BASTIN_SOLVER_TRADITIONAL_HPP

