#ifndef KUBO_BASTIN_SOLVER_HPP
#define KUBO_BASTIN_SOLVER_HPP

#include<string>
#include"static_vars.hpp"
#include "Graphene.hpp"
#include "kernel.hpp"
#include "vec_base.hpp"

struct solver_vars{
  
  type a_ ,b_, E_min_, eta_, E_start_, E_end_, edge_;//m_str, rsh_str, anderson_str;
  int M_, R_;
  std::string filename_, run_dir_;
  
};



class KuboBastin_solver{

private:
  solver_vars parameters_;
  Graphene& device_;
  Kernel* kernel_;
public:
  ~KuboBastin_solver(){delete kernel_;};
  KuboBastin_solver();
  KuboBastin_solver( solver_vars&, Graphene&);
  
  solver_vars& parameters(){return parameters_;};
  void compute();


  void integration(type*, type*, type* );
  void update_data(type*, type*, type*, type*, int ,  std::string, std::string );
  void plot_data(std::string, std::string );
  void polynomial_cycle(type*, type*, type*, type*, type*,  type , type );
  void polynomial_cycle_ket(type*, type*, type*, type*, type*,  type , type );
  void KuboBastin_FFTs(type*, type*, type*, type*);
  
};


#endif //KUBO_BASTIN_SOLVER_HPP
