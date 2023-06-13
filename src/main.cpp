#include<iostream>
#include<fstream>
#include<string>

#include "static_vars.hpp"
#include "Device/Device.hpp"
#include "Device/Graphene.hpp"
#include "Device/TBG/TBG.hpp"
#include "Kubo_solver/Kubo_solver.hpp"


int main(int , char **argv){

  solver_vars s_vars;
  device_vars graphene_vars;
  
  std::ifstream Input;
  Input.open(argv[1]);
  
  Input>>s_vars.run_dir_;


  int device_choice=0;
  Input>>device_choice;
  //Reading device variables
  Input>>graphene_vars.W_,  Input>>graphene_vars.LE_,  Input>>graphene_vars.C_, Input>>graphene_vars.theta_, Input>>graphene_vars.d_min_;
  Input>>graphene_vars.dis_str_, Input>>graphene_vars.dis_seed_;
  Input>>s_vars.cap_choice_;

  graphene_vars.theta_*= 2.0*M_PI/360.0;
  graphene_vars.SUBDIM_ = graphene_vars.W_*graphene_vars.LE_;
  graphene_vars.DIM_    = graphene_vars.SUBDIM_ + 2*graphene_vars.C_*graphene_vars.W_;



  //Reading simulation parameters
  Input>>s_vars.base_choice_;
  Input>>s_vars.kernel_choice_;
  Input>>s_vars.M_, Input>>s_vars.R_, Input>>s_vars.dis_real_, Input>>s_vars.seed_, Input>>s_vars.edge_,
  Input>>s_vars.num_parts_,   Input>>s_vars.num_p_;
  Input>>s_vars.E_start_,   Input>>s_vars.E_end_;
  Input>>s_vars.eta_;
  Input>>s_vars.E_min_;
  Input>>s_vars.filename_;


  s_vars.SECTION_SIZE_ = graphene_vars.SUBDIM_/s_vars.num_parts_;

  


  s_vars.a_ = 1.0;
  s_vars.b_ = 0.0;

  //  double Eedge = 8.09852;
  //s_vars.a_ = 2*Eedge/(2.0-s_vars.edge_);

  if(s_vars.num_p_<s_vars.M_)
    s_vars.num_p_=s_vars.M_;

  Device *device;

  if(device_choice==0)
    device = new Graphene(graphene_vars);
  if(device_choice==1)
    device = new TBG(graphene_vars);
  
  Kubo_solver solver( s_vars, *device);
  solver.compute();

  return 0;
}

