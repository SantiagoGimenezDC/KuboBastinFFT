#ifndef KUBO_BASTIN_SOLVER_HPP
#define KUBO_BASTIN_SOLVER_HPP

#include<string>
#include"static_vars.hpp"


struct solver_vars{
  
  type a_ ,b_, E_min_, eta_, E_start_, E_end_, edge_;//m_str, rsh_str, anderson_str;
  int R_;
  std::string filename_, run_dir_;
  
};


void integration(type*, type*, type* );
void update_data(type*, type*, type*, int , type , std::string, std::string );
void plot_data(std::string, std::string );
void polynomial_cycle(type*, type*, type*, type*, type*,  type , type );
void KuboBastin_solver(solver_vars& );
void KuboBastin_FFTs(type*, type*, type*, type*);


#endif //KUBO_BASTIN_SOLVER_HPP
