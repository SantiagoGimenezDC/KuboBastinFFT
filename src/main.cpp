#include<iostream>
#include<memory>
#include<string>

#include "static_vars.hpp"
#include "Input/Input.hpp"
#include "Device/Device.hpp"

#include "KPM_base/KPM_base.hpp"
#include "Kubo_solver/Kubo_solver_FFT/Kubo_solver_FFT.hpp"
#include "Kubo_solver/Kubo_solver_filtered/Kubo_solver_filtered.hpp"
#include "Kubo_solver/Kubo_solver_traditional/Kubo_solver_traditional.hpp"


int main(int argc, char **argv){

  if( argc < 2 ){
    std::cerr<<"Usage: "<<argv[0]<<" <input.toml>"<<std::endl;
    return 1;
  }

  input_config cfg;
  std::unique_ptr<Device> device;  // freed automatically when main() returns

  try{
    cfg    = read_input(argv[1]);
    device = make_device(cfg);
  }
  catch(const input_error& err){
    std::cerr<<"Input error: "<<err.what()<<std::endl;
    return 1;
  }


  std::cout<<"Ongoing gimmicks:"<<std::endl;
  std::cout<<"AUTO BOUND DETECTION NOT WORKING - on normal mode only???"<<std::endl;
  std::cout<<"Read_Hamiltonian only works for COMPLEX Hamiltonian;"<<std::endl;
  std::cout<<"Min max eigenvalues is using H_ket with the 4 entries;"<<std::endl;
  std::cout<<"The min max eigv are fixed;"<<std::endl;


  solver_vars& s_vars = cfg.s_vars;

  if( cfg.sim_type == "DOS" ){
    KPM_DOS_solver solver( s_vars, *device );
    solver.compute();
  }

  if( cfg.sim_type == "normal" ){
    Kubo_solver_FFT solver( s_vars, *device );
    solver.compute();
  }

  if( cfg.sim_type == "traditional" ){
    Kubo_solver_traditional solver( s_vars, *device );
    solver.compute();
  }

  if( cfg.sim_type == "filtered" ){
    KB_filter filter( cfg.f_vars );
    Kubo_solver_filtered solver( s_vars, *device, filter );
    solver.compute();
  }

  return 0;
}
