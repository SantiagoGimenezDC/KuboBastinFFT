#include<iostream>
#include<fstream>
#include<string>
#include<iomanip>
#include<cstdlib>

#include<complex>



#include "static_vars.hpp"
#include "Graphene.hpp"
#include "KuboBastin_solver.hpp"






int main(int , char **argv){

  solver_vars s_vars;
  device_vars graphene_vars;
  
  std::ifstream Input;
  Input.open(argv[1]);
  
  Input>>s_vars.run_dir_;


  //Reading device variables
  Input>>graphene_vars.W_,  Input>>graphene_vars.LE_,  Input>>graphene_vars.C_;

  graphene_vars.SUBDIM_ = graphene_vars.W_*graphene_vars.LE_;
  graphene_vars.DIM_    = graphene_vars.SUBDIM_ + 2*graphene_vars.C_*graphene_vars.W_;

  

  //Reading simulation parameters
  Input>>s_vars.M_, Input>>s_vars.R_, Input>>s_vars.edge_,
  Input>>s_vars.E_start_,   Input>>s_vars.E_end_;
  Input>>s_vars.eta_;
  Input>>s_vars.E_min_;
  Input>>s_vars.filename_;

  
  s_vars.a_= 10;
  s_vars.b_ = 0;


  Graphene graphene_device(graphene_vars);
  
  KuboBastin_solver solver( s_vars, graphene_device);
  solver.compute();
  
}

