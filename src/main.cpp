#include<iostream>
#include<fstream>
#include<string>

#include "static_vars.hpp"
#include "Device/Device.hpp"
#include "Device/Graphene.hpp"
#include "Device/TBG/TBG.hpp"
#include "Kubo_solver/Kubo_solver.hpp"
#include "Kubo_solver/Kubo_solver_SSD.hpp"
#include "Kubo_solver/Kubo_solver_filtered.hpp"

int main(int , char **argv){

  solver_vars s_vars;
  device_vars graphene_vars;
  
  std::ifstream Input;
  Input.open(argv[1]);
  
  Input>>s_vars.run_dir_;


  int device_choice=0;
  double   RAM_size=0; //in GB
  std::string sim_type;
  
  Input>>device_choice;
  //Reading device variables
  Input>>graphene_vars.W_,  Input>>graphene_vars.LE_,  Input>>graphene_vars.C_, Input>>graphene_vars.theta_, Input>>graphene_vars.d_min_;
  Input>>graphene_vars.dis_str_, Input>>graphene_vars.dis_seed_;
  Input>>s_vars.cap_choice_;

  graphene_vars.theta_*= 2.0 * M_PI/360.0;
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

  Input>>sim_type;
  Input>>RAM_size;

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

  
  if(sim_type == "SSD"){
    Kubo_solver_SSD solver( s_vars, RAM_size,  *device);
    solver.compute();
  }
  if(sim_type == "normal"){
    Kubo_solver solver( s_vars, *device);
    solver.compute();
  }
  if(sim_type == "filtered"){
    
    filter_vars f_vars;

    f_vars.M_ = s_vars.M_;
    Input>>f_vars.post_filter_;
    Input>>f_vars.filter_;
    Input>>f_vars.L_,   Input>>f_vars.decRate_;
    Input>>f_vars.k_dis_, Input>>f_vars.f_cutoff_;
    Input>>f_vars.att_;

    if(!f_vars.post_filter_ && !f_vars.filter_)
      f_vars.decRate_=1;

    f_vars.M_ext_ =  f_vars.M_;
    f_vars.k_dis_ += f_vars.M_ext_/4;
    f_vars.nump_  = f_vars.M_ext_/f_vars.decRate_;

    s_vars.num_p_ = f_vars.nump_;

    
    //f_vars.k_dis_ = f_vars.M_/4;
    //f_vars.f_cutoff_ = f_vars.M_/ 30; 
    //f_vars.att_ = 96;

    
    KB_filter filter(f_vars); 
    Kubo_solver_filtered solver( s_vars, *device, filter);
    solver.compute();
  }
  
  return 0;
}

