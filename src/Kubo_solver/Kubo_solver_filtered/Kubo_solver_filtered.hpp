
#ifndef KUBO_BASTIN_FILTERED_SOLVER_HPP
#define KUBO_BASTIN_FILTERED_SOLVER_HPP

#include<string>
#include"../../static_vars.hpp"
#include "../../Device/Device.hpp"
#include "../../Device/Graphene.hpp"
#include "../../Device/ArmchairGraph_RashbaSOC.hpp"
#include "../../kernel.hpp"
#include "../../vec_base.hpp"
#include "../../CAP.hpp"

#include "KB_filter.hpp"
#include "../solver_vars.hpp"

/*
struct solver_vars{  
  r_type a_ ,b_, E_min_, eta_, E_start_, E_end_, edge_;//m_str, rsh_str, anderson_str;
  int M_, R_, nump_, dis_real_, seed_, num_parts_, SECTION_SIZE_;
  std::string filename_, run_dir_;
  int cap_choice_, base_choice_, kernel_choice_;  
};
*/


class Kubo_solver_filtered{
private:
  solver_vars parameters_;
  Device&  device_;
  KB_filter& filter_;
  formula sym_formula_;

  Kernel*   kernel_;
  CAP*      cap_;
  Vec_Base* vec_base_;

  const double initial_disp_ = 0.5;
  
public:
  ~Kubo_solver_filtered(){delete kernel_, delete cap_, delete vec_base_;};
  Kubo_solver_filtered();
  Kubo_solver_filtered( solver_vars&, Device&, KB_filter&);
  
  solver_vars& parameters(){return parameters_;};

  
  void interpolated_integration(const r_type* , const r_type* , r_type* );
  
  void reset_buffer(type**);
  
  void compute(){
          compute_imag();/*
    if( parameters_.base_choice_ == 1  )
      compute_imag();
    else
    compute_real();*/
  };


  void batch_vel_op(std::complex<r_type>**, int, int );

  
  //For REAL randVECs
  void compute_real();  

  void filter( int, type*, type**, type*, type*, int, int);
  void filter_2( int, type*, type**, type*, type*, int, int);
  void filter_2_doubleBuffer( int, type*, type**, type**, type*, type*, int, int);
  
  void filtered_polynomial_cycle_OTF( type** , type*,  r_type* , r_type* , int , int );
  void filtered_polynomial_cycle_direct( type** , type*, int , int );
  void filtered_polynomial_cycle_direct_doubleBuffer( type** , type**, type*, int , int );
  
  void Greenwood_FFTs( std::complex<r_type>**, std::complex<r_type>**,  type*, int);
  void Bastin_FFTs   ( r_type*, std::complex<r_type>**, std::complex<r_type>**,  type*, int );
  void Sea_FFTs   ( r_type*, std::complex<r_type>**, std::complex<r_type>**,  type*, int );  

  void Bastin_FFTs_doubleBuffer   ( r_type*, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**, type*, int );


  


  


  //For IMAG randVECs  
  void compute_imag();
  
  void filter_imag( int, type*, type**, type**, type*, type*, int, int);
  void filter_doubleBuffer_imag( int, type*, type**, type**, type**, type**, type*, type*, int, int);

  void filtered_polynomial_cycle_OTF_imag( type** , type** ,  type*, int , int );  
  void filtered_polynomial_cycle_direct_imag( type** , type** , type*, int , int );
  void filtered_polynomial_cycle_direct_doubleBuffer_imag( type** , type**, type** , type** , type*, int , int );
  
  void Greenwood_FFTs_imag( std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**,  type*, int );
  void Bastin_FFTs_imag   ( r_type*, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**,  type*, int );
   void Sea_FFTs_imag   ( r_type*, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**,  type*, int );
  
  void Bastin_FFTs_doubleBuffer_imag   ( r_type*, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**,  type*, int );
  void Sea_FFTs_doubleBuffer_imag   ( r_type*, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**, std::complex<r_type>**,  type*, int );

  
  

  //*POST-PROCESSING------------------

  void rearrange_crescent_order( r_type* );
  void integration_linqt(const r_type* , const r_type* , r_type* );
  
  void compute_E_points( r_type* );

  void integration ( r_type*, r_type*, r_type* );
  void eta_CAP_correct(r_type*, r_type* );
  void update_data ( r_type*, type*, type*, r_type*, int ,  std::string, std::string );
  void update_data_Bastin ( r_type*, type*, type*, r_type*,  int ,  std::string, std::string );
  void update_data_Sea ( r_type*, type*, type*, r_type*,  int ,  std::string, std::string );

  void plot_data   ( std::string, std::string );
  //*POST-PROCESSING------------------

  
inline
void copy_vector(type vec_destination[], type vec_original[], int size){
#pragma omp parallel for
  for(int i=0;i<size;i++)
    vec_destination[i] = vec_original[i];
}

inline
void plus_eq(type vec_1[], type vec_2[], type factor, int size){
#pragma omp parallel for
  for(int i=0;i<size;i++)
    vec_1[i] += factor * vec_2[i];
}
  
inline
void plus_eq_imag(type vec_1_re[], type vec_1_im[], type vec_2[], type factor, int size){
#pragma omp parallel for
  for(int i=0;i<size;i++){
    vec_1_re[i] += factor * real ( vec_2[i] );
    vec_1_im[i] += factor * imag ( vec_2[i] );
  }
}
  
inline
void ay(type factor, type vec[], int size){
#pragma omp parallel for
  for(int i=0;i<size;i++)
    vec[i] * factor;
}


};


#endif //KUBO_BASTIN_FILTERED_SOLVER_HPP
