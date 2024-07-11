#ifndef KUBO_BASTIN_FILTERED_SOLVER_HPP
#define KUBO_BASTIN_FILTERED_SOLVER_HPP

#include<string>
#include"../static_vars.hpp"
#include "../Device/Device.hpp"
#include "../Device/Graphene.hpp"
#include "../kernel.hpp"
#include "../vec_base.hpp"

#include "KB_filter.hpp"
#include "solver_vars.hpp"

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
  
  Kernel*   kernel_;
  CAP*      cap_;
  Vec_Base* vec_base_;

  const double initial_disp_ = 0.5;
  
public:
  ~Kubo_solver_filtered(){delete kernel_, delete cap_, delete vec_base_;};
  Kubo_solver_filtered();
  Kubo_solver_filtered( solver_vars&, Device&, KB_filter&);
  
  solver_vars& parameters(){return parameters_;};
  void compute();
  void compute_E_points( r_type* );

  void integration ( r_type*, r_type*, r_type* );
  void eta_CAP_correct(r_type*, r_type* );
  void update_data ( r_type*,r_type*, r_type*, r_type*, r_type*, int ,  std::string, std::string );
  void plot_data   ( std::string, std::string );


  void reset_buffer(type**);
  void filter( int, type*, type**, type*, type*, int, int);
  void filter_2( int, type*, type**, type*, type*, int, int);
  
  void filtered_polynomial_cycle( type** , type*,  r_type* , r_type* , int , int );
  void filtered_polynomial_cycle_direct( type** , type*,  r_type* , r_type* , int , int );
  void filtered_polynomial_cycle_direct_2( type** , type*, int , int );
  
  //  void Bastin_FFTs__reVec_noEta     ( r_type*, r_type*, r_type*, r_type*);
  //void Bastin_FFTs__imVec_noEta     ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);
  //void Bastin_FFTs__imVec_eta       ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);

  //void Greenwood_FFTs__reVec_noEta ( r_type**, r_type**, r_type*, r_type*);  
  //void Greenwood_FFTs__reVec_eta   ( r_type*, r_type*, r_type*, r_type*);
  void Greenwood_FFTs( std::complex<r_type>**, std::complex<r_type>**,  r_type*);
  void Greenwood_FFTs_2( std::complex<r_type>**, std::complex<r_type>**,  r_type*);
  void Bastin_FFTs   ( std::complex<r_type>**, std::complex<r_type>**,  r_type*);
  
};


#endif //KUBO_BASTIN_FILTERED_SOLVER_HPP
