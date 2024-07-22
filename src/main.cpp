#include<iostream>
#include<fstream>
#include<string>


#include "static_vars.hpp"
#include "Device/Device.hpp"
#include "Device/Graphene.hpp"
#include "Device/Read_Hamiltonian.hpp"
#include "Device/Read_ConTable.hpp"

#include "Device/TBG/TBG.hpp"
#include "Kubo_solver/Kubo_solver_FFT/Kubo_solver_FFT.hpp"
#include "Kubo_solver/Kubo_solver_SSD.hpp"
#include "Kubo_solver/Kubo_solver_filtered.hpp"
#include "Kubo_solver/Kubo_solver_traditional/Kubo_solver_traditional.hpp"

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
  graphene_vars.DIM_    = graphene_vars.SUBDIM_ + 2 * graphene_vars.C_*graphene_vars.W_;



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

  Input>>graphene_vars.filename_;
  graphene_vars.run_dir_ = s_vars.run_dir_;

  
  /*  
  Read_ConTable test_con(graphene_vars);
  Read_Hamiltonian test_ham(graphene_vars);
 
  test_ham.build_Hamiltonian();
  test_con.build_Hamiltonian();
  test_ham.setup_velOp();
  test_con.setup_velOp();

 
  for(int i=0;i<test_con.H().nonZeros(); i++){
    double error = abs ( test_con.H().valuePtr()[i] - test_ham.H().valuePtr()[i] ) /  test_ham.H().valuePtr()[i] ;
    if( error > 0.00001)
      std::cout<<i<<"  "<<test_con.H().valuePtr()[i]<<"    "<<test_ham.H().valuePtr()[i]<<std::endl;

  }
    std::cout<<"Nonzeros: "<<test_con.vx().nonZeros()<<"  "<<test_ham.vx().nonZeros()<<std::endl;
  std::cout<<(test_con.vx()-test_ham.vx()).norm()<<std::endl;
  */
  
  if(device_choice==0)
    device = new Graphene(graphene_vars);
  if(device_choice==1)
    device = new TBG(graphene_vars);
  if(device_choice==2)
    device = new Read_Hamiltonian(graphene_vars);
  if(device_choice==3)
    device = new Read_ConTable(graphene_vars);


  if(sim_type == "SSD"){
    Kubo_solver_SSD solver( s_vars, RAM_size,  *device);
    solver.compute();
  }

  if(sim_type == "normal"){
    Kubo_solver_FFT solver( s_vars, *device);
    solver.compute();
  }

  if(sim_type == "traditional"){
    Kubo_solver_traditional solver( s_vars, *device);
    solver.compute();
  }


  if(sim_type == "filtered"){

    filter_vars f_vars;

    f_vars.M_ = s_vars.M_;
    Input>>f_vars.post_filter_;
    Input>>f_vars.filter_;
    Input>>f_vars.L_,   Input>>f_vars.decRate_;
    Input>>f_vars.energy_center_, Input>>f_vars.f_cutoff_;
    Input>>f_vars.att_;

    if(!f_vars.post_filter_ && !f_vars.filter_)
      f_vars.decRate_ = 1;

    
    f_vars.M_ext_ =  2*f_vars.M_ ;
    f_vars.k_dis_ = f_vars.M_ext_/4;
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


  /*
  Read_Hamiltonian test_ham(graphene_vars);
  Read_ConTable test_con(graphene_vars);

  test_ham.build_Hamiltonian();
  test_con.build_Hamiltonian();
  test_ham.setup_velOp();
  test_con.setup_velOp();


  for(int i=0;i<test_con.H().nonZeros(); i++){
    if(test_con.H().valuePtr()[i]-test_ham.H().valuePtr()[i] != 0 && i < 400)
      std::cout<<i<<"  "<<test_con.H().valuePtr()[i]<<"    "<<test_ham.H().valuePtr()[i]<<std::endl;

  }
    std::cout<<"Nonzeros: "<<test_con.vx().nonZeros()<<"  "<<test_ham.vx().nonZeros()<<std::endl;
  std::cout<<(test_con.vx()-test_ham.vx()).norm()<<std::endl;
*/
  
