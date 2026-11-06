#include <cmath>
#include <iostream>
#include <vector>

#include "Input.hpp"

#include "../Device/Graphene.hpp"
#include "../Device/Graphene_KaneMele.hpp"
#include "../Device/Graphene_supercell.hpp"
#include "../Device/ArmchairGraph_RashbaSOC.hpp"
#include "../Device/SupercellGraph_RashbaSOC.hpp"
#include "../Device/Read_Hamiltonian.hpp"
#include "../Device/Read_Siesta.hpp"
#include "../Device/Read_Siesta_Cropped.hpp"
#include "../Device/Read_ConTable.hpp"
#include "../Device/TBG/TBG.hpp"
#include "../Device/Graphene_nanoribbon_kspace.hpp"



static bool ends_with(const std::string& s, const std::string& suffix){
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}



input_config read_input(const std::string& path){
  input_config cfg;

  if( ends_with(path, ".toml") )
    cfg = read_toml_input(path);
  else{
    std::cout<<"WARNING: '"<<path<<"' is read as a legacy positional input file."<<std::endl
	     <<"         This format is deprecated; please switch to a .toml input (see examples/template/input.toml)."<<std::endl;
    cfg = read_legacy_input(path);
  }

  cfg.s_vars.para_file_ = path;
  finalize_input(cfg);
  return cfg;
}



void finalize_input(input_config& cfg){
  solver_vars& s = cfg.s_vars;
  device_vars& d = cfg.d_vars;
  filter_vars& f = cfg.f_vars;

  d.run_dir_ = s.run_dir_;
  d.theta_  *= 2.0 * M_PI / 360.0;  // input in degrees
  d.SUBDIM_  = d.W_ * d.LE_;
  d.DIM_     = d.SUBDIM_ + 2 * d.C_ * d.W_;

  s.num_p_        = s.M_ * cfg.nump_factor;
  s.SECTION_SIZE_ = d.SUBDIM_ / s.num_parts_;


  if( cfg.sim_type == "filtered" ){
    f.M_     = s.M_;
    f.M_ext_ = cfg.nump_factor * f.M_;

    if( f.decRate_ == 1 ){
      f.L_        = 1;
      f.f_cutoff_ = f.M_ext_ * 2;
      f.nump_     = f.M_ext_;  // no decimation: keep every point
    }
    else{
      f.L_        = cfg.L_factor * f.decRate_ + ( int(cfg.L_factor * f.decRate_) % 2 == 0 ? 1 : 0 ); //L_fact=40 is a good default
      f.f_cutoff_ = cfg.cutoff_factor * f.M_ext_ / ( 2.0 * f.decRate_ ); //a default estimate of the cutoff. Verify. Could be greedier for Greenwood
      f.nump_     = f.M_ext_ / f.decRate_;
    }

    s.num_p_ = f.nump_;
  }
}



static const std::vector<std::string> device_names = {
  "Graphene",                    // 0
  "TBG",                         // 1
  "Read_Hamiltonian",            // 2
  "Read_ConTable",               // 3
  "ArmchairGraph_RashbaSOC",     // 4
  "Graphene_supercell",          // 5
  "SupercellGraph_RashbaSOC",    // 6
  "Graphene_KaneMele",           // 7
  "Graphene_nanoribbon_kspace",  // 8
  "Read_Siesta",                 // 9
  "Read_Siesta_Cropped"          // 10
};


const std::string& device_name(int legacy_choice){
  if( legacy_choice < 0 || legacy_choice >= int(device_names.size()) )
    throw input_error("device_choice " + std::to_string(legacy_choice) + " is out of range (0-"
		      + std::to_string(device_names.size() - 1) + ")");
  return device_names[legacy_choice];
}



std::unique_ptr<Device> make_device(input_config& cfg){
  device_vars&      d = cfg.d_vars;
  const model_vars& m = cfg.m_vars;
  const std::string& type = cfg.device_type;

  if( type == "Graphene" )                   return std::make_unique<Graphene>(d);
  if( type == "TBG" )                        return std::make_unique<TBG>(d);
  if( type == "Read_Hamiltonian" )           return std::make_unique<Read_Hamiltonian>(d);
  if( type == "Read_ConTable" )              return std::make_unique<Read_ConTable>(d);
  if( type == "ArmchairGraph_RashbaSOC" )    return std::make_unique<ArmchairGraph_RashbaSOC>(m.m_str_, m.rashba_str_, d);
  if( type == "Graphene_supercell" )         return std::make_unique<Graphene_supercell>(d);
  if( type == "SupercellGraph_RashbaSOC" )   return std::make_unique<SupercellGraph_RashbaSOC>(m.m_str_, m.rashba_str_, d);
  if( type == "Graphene_KaneMele" )          return std::make_unique<Graphene_KaneMele>(m.k_space_, m.range_, m.stgr_str_, m.m_str_, m.rashba_str_, m.KM_str_, m.HLD_str_, d);
  if( type == "Graphene_nanoribbon_kspace" ) return std::make_unique<Graphene_nanoribon_kspace>(m.k_space_, m.range_, m.stgr_str_, m.m_str_, m.rashba_str_, m.KM_str_, m.HLD_str_, d);
  if( type == "Read_Siesta" )                return std::make_unique<Read_Siesta>(d);
  if( type == "Read_Siesta_Cropped" )        return std::make_unique<Read_Siesta_Cropped>(d);

  throw input_error("unknown device type '" + type + "'");
}
