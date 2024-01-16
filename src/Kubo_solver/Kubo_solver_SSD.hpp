#ifndef KUBO_BASTIN_SOLVER_SSD_HPP
#define KUBO_BASTIN_SOLVER_SSD_HPP

#include<string>
#include"../static_vars.hpp"
#include "../Device/Device.hpp"
#include "../Device/Graphene.hpp"
#include "../kernel.hpp"
#include "../vec_base.hpp"

#include<eigen-3.4.0/Eigen/Core>

#include "Kubo_solver.hpp" //For the input struct;
/*
struct solver_vars{  
  r_type a_ ,b_, E_min_, eta_, E_start_, E_end_, edge_;//m_str, rsh_str, anderson_str;
  int M_, R_, dis_real_, seed_, num_parts_, num_p_, SECTION_SIZE_;
  std::string filename_, run_dir_;
  int cap_choice_, base_choice_, kernel_choice_;  
};
*/


class Kubo_solver_SSD{
private:
  solver_vars parameters_;
  Device&  device_;
  
  Kernel*   kernel_;
  CAP*      cap_;
  Vec_Base* vec_base_;

  double RAM_buffer_size_;//In GBs
  int num_buffers_;
public:
  ~Kubo_solver_SSD(){delete kernel_, delete cap_, delete vec_base_;};
  Kubo_solver_SSD();
  Kubo_solver_SSD( solver_vars&, Device&);
  
  solver_vars& parameters(){return parameters_;};
  void compute();


  void integration ( r_type*, r_type*, r_type* );
  void eta_CAP_correct(r_type*, r_type* );
  void update_data ( r_type*,r_type*, r_type*, r_type*, r_type*, int ,  std::string, std::string );
  void plot_data   ( std::string, std::string );
  
  void polynomial_cycle     ( type*, type*, type*, type*, r_type*, r_type* , int);
  void polynomial_cycle_ket ( type*, type*, type*, type*, r_type*, r_type* , int);
  
  void Greenwood_FFTs__imVec_noEta_SSD ( std::complex<r_type>*, std::complex<r_type>*, r_type*, r_type*);

  
  void create_buffers();
  void transfer_to_buffer(const std::string &, type*);
  void transfer_to_SSD(const std::string &, type*);
  void update_SSD_buffer( const std::string &, const int, type *);

};


#endif //KUBO_BASTIN_SOLVER_HPP
