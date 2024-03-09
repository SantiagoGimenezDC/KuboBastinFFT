#ifndef KUBO_BASTIN_SOLVER_HPP
#define KUBO_BASTIN_SOLVER_HPP

#include<string>
#include<iostream>
#include<eigen-3.4.0/Eigen/Core>

#include"../../static_vars.hpp"
#include "../../Device/Device.hpp"
#include "../../Device/Graphene.hpp"
#include "../../kernel.hpp"
#include "../../vec_base.hpp"

#include "../solver_vars.hpp"


class Kubo_solver_FFT{
private:
  solver_vars parameters_;
  formula sym_formula_;

  
  Device&  device_;
  Kernel*   kernel_;
  CAP*      cap_;
  Vec_Base* vec_base_;

  

  
//---------------Large vectors----------------//  
  type **bras_ ,
       **kets_ ;
  
  //Recursion Vectors
  type *vec_,
       *p_vec_,
       *pp_vec_,
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
  Kubo_solver_FFT();
  Kubo_solver_FFT( solver_vars&, Device&);
  ~Kubo_solver_FFT();
  
  solver_vars& parameters(){ return parameters_; };

  //Initializers
  void allocate_memory();
  void reset_recursion_vectors();
  void reset_Chebyshev_buffers();


  //Heavy duty
  void compute();

  void polynomial_cycle     ( type**, int);
  void polynomial_cycle_ket ( type**, int);

  void Greenwood_FFTs__imVec ( type**, type**,  type*, int);
  void Bastin_FFTs ( type**, type**, type*, int);



  
  //Post-process
  void Greenwood_postProcess (  int  );
  void Bastin_postProcess ( int  );

  void integration ( r_type*, r_type*, r_type* );
  void partial_integration ( r_type*, r_type*, r_type* );

  void rearrange_crescent_order( r_type* );
  void eta_CAP_correct(r_type*, r_type* );  
  void plot_data   ( std::string, std::string );
  
};


#endif //KUBO_BASTIN_SOLVER_HPP


/*
  
  void Bastin_FFTs__reVec_noEta_2     ( r_type**, r_type**, r_type*, r_type*);
  void Bastin_FFTs__imVec_noEta_2 ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);
  
  void Bastin_FFTs__reVec_noEta     ( r_type**, r_type**, r_type*, r_type*);
  void Bastin_FFTs__imVec_noEta     ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);
  void Bastin_FFTs__imVec_eta       ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);

  void Greenwood_FFTs__reVec_noEta ( r_type**, r_type**, r_type*, r_type*);  
  void Greenwood_FFTs__reVec_eta   ( r_type*, r_type*, r_type*, r_type*);
  
  void Greenwood_FFTs__imVec_eta ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);




*/
