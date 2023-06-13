#ifndef KUBO_BASTIN_SOLVER_HPP
#define KUBO_BASTIN_SOLVER_HPP

#include<string>
#include"../static_vars.hpp"
#include "../Device/Device.hpp"
#include "../Device/Graphene.hpp"
#include "../kernel.hpp"
#include "../vec_base.hpp"

#include<eigen-3.4.0/Eigen/Core>

struct solver_vars{  
  r_type a_ ,b_, E_min_, eta_, E_start_, E_end_, edge_;//m_str, rsh_str, anderson_str;
  int M_, R_, dis_real_, seed_, num_parts_, num_p_, SECTION_SIZE_;
  std::string filename_, run_dir_;
  int cap_choice_, base_choice_, kernel_choice_;  
};



class Kubo_solver{
private:
  solver_vars parameters_;
  Device&  device_;
  
  Kernel*   kernel_;
  CAP*      cap_;
  Vec_Base* vec_base_;

  
public:
  ~Kubo_solver(){delete kernel_, delete cap_, delete vec_base_;};
  Kubo_solver();
  Kubo_solver( solver_vars&, Device&);
  
  solver_vars& parameters(){return parameters_;};
  void compute();


  void integration ( r_type*, r_type*, r_type* );
  void eta_CAP_correct(r_type*, r_type* );
  void update_data ( r_type*,r_type*, r_type*, r_type*, r_type*, int ,  std::string, std::string );
  void plot_data   ( std::string, std::string );
  
  void polynomial_cycle     ( type**, type*, type*, type*, r_type*, r_type* , int);
  void polynomial_cycle_ket ( type**, type*, type*, type*, r_type*, r_type* , int);


  
  void Bastin_FFTs__reVec_noEta_2     ( r_type**, r_type**, r_type*, r_type*);
  void Bastin_FFTs__imVec_noEta_2 ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);
  
  void Bastin_FFTs__reVec_noEta     ( r_type**, r_type**, r_type*, r_type*);
  void Bastin_FFTs__imVec_noEta     ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);
  void Bastin_FFTs__imVec_eta       ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);

  void Greenwood_FFTs__reVec_noEta ( r_type**, r_type**, r_type*, r_type*);  
  void Greenwood_FFTs__reVec_eta   ( r_type*, r_type*, r_type*, r_type*);
  void Greenwood_FFTs__imVec_noEta ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);
  void Greenwood_FFTs__imVec_eta ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);



  
  void StandardProcess_Bastin(std::complex<r_type> **, std::complex<r_type> **, r_type*, r_type* );
  void StandardProcess_Greenwood(std::complex<r_type> **, std::complex<r_type> **, r_type*, r_type* );  
  Eigen::Matrix<type, -1,-1, Eigen::ColMajor> fill_green(r_type*, int , int );
  type green(int , type );
  type dgreen(int , type );
  Eigen::Matrix<type, -1,-1, Eigen::RowMajor> fill_dgreen(r_type*, int , int );
  
};


#endif //KUBO_BASTIN_SOLVER_HPP
