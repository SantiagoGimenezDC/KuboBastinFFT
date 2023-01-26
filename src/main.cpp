#include<iostream>
#include<fstream>
#include<string>
#include<iomanip>
#include<cstdlib>

#include<complex>



#include "static_vars.hpp"

#include "test_hamiltonian.hpp"
#include "KuboBastin_solver.hpp"






int main(int , char **argv){

  solver_vars s_vars;

  
  std::ifstream Input;
  Input.open(argv[1]);
  

  Input>>s_vars.run_dir_;
  Input>>s_vars.R_, Input>>s_vars.edge_,
  Input>>s_vars.E_start_,   Input>>s_vars.E_end_;
  Input>>s_vars.eta_;
  Input>>s_vars.E_min_;
  Input>>s_vars.filename_;

  
  s_vars.a_= 10;
  s_vars.b_ = 0;

  
  KuboBastin_solver(s_vars);
  
  
}

