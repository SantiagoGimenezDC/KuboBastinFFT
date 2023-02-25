#include<iostream>
#include<fstream>
#include<string>

#include "static_vars.hpp"
#include "Graphene.hpp"
#include "Kubo_solver/Kubo_solver.hpp"


int main(int , char **argv){

  solver_vars s_vars;
  device_vars graphene_vars;
  
  std::ifstream Input;
  Input.open(argv[1]);
  
  Input>>s_vars.run_dir_;


  //Reading device variables
  Input>>graphene_vars.W_,  Input>>graphene_vars.LE_,  Input>>graphene_vars.C_;
  Input>>graphene_vars.dis_str_, Input>>graphene_vars.dis_seed_;
  
  graphene_vars.SUBDIM_ = graphene_vars.W_*graphene_vars.LE_;
  graphene_vars.DIM_    = graphene_vars.SUBDIM_ + 2*graphene_vars.C_*graphene_vars.W_;

  

  //Reading simulation parameters
  Input>>s_vars.M_, Input>>s_vars.R_, Input>>s_vars.dis_real_, Input>>s_vars.seed_, Input>>s_vars.edge_,
  Input>>s_vars.num_parts_;
  Input>>s_vars.E_start_,   Input>>s_vars.E_end_;
  Input>>s_vars.eta_;
  Input>>s_vars.E_min_;
  Input>>s_vars.filename_;


  s_vars.SECTION_SIZE_ = graphene_vars.SUBDIM_/s_vars.num_parts_;

  
  if(graphene_vars.SUBDIM_%s_vars.num_parts_!=0){
    std::cout<<"Please select SUBDIM to be divisible by num_parts_"<<std::endl;
    return 0;
  }

  

  Graphene graphene_device(graphene_vars);

  int maxIter = 300;
  r_type Emax, Emin,  edge = s_vars.edge_, Eedge;
  
  graphene_device.minMax_EigenValues( maxIter,  Emax, Emin);
  Eedge=std::max(std::abs(Emax), std::abs(Emin));

  s_vars.a_ = 2*Eedge/(2.0-edge);
  s_vars.b_ = 0;


  
  Kubo_solver solver( s_vars, graphene_device);
  solver.compute();

  return 0;
}

