#ifndef SOLVER_VARS_HPP
#define SOLVER_VARS_HPP

#include<string>
#include"../static_vars.hpp"



enum formula{
  KUBO_GREENWOOD = 0,
  KUBO_BASTIN = 1,
  KUBO_SEA = 2
};


struct solver_vars{  
  r_type a_ ,b_, E_min_, eta_, E_start_, E_end_, edge_;//m_str, rsh_str, anderson_str;
  int M_, R_,  dis_real_, seed_, num_parts_, num_p_,SECTION_SIZE_;
  std::string filename_, run_dir_;
  int cap_choice_, base_choice_, kernel_choice_, vel_dir_1_, vel_dir_2_;  

  formula sim_equation_;
};

void Station(int , std::string  );

#endif //SOLVER_VARS_HPP
