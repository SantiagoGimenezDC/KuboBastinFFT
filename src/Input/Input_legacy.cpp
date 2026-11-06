// Reader for the old positional input format (SimData.dat style).
// Deprecated: kept so existing runs still work. New inputs should use TOML (Input_toml.cpp).

#include <fstream>

#include "Input.hpp"


input_config read_legacy_input(const std::string& path){

  std::ifstream Input(path);
  if( !Input )
    throw input_error("cannot open input file '" + path + "'");

  input_config cfg;
  solver_vars& s_vars        = cfg.s_vars;
  device_vars& graphene_vars = cfg.d_vars;
  filter_vars& f_vars        = cfg.f_vars;
  model_vars&  m_vars        = cfg.m_vars;

  int device_choice = 0;
  double RAM_size   = 0; //in GB, unused (SSD solver is disabled)
  std::string sim_equation, string_is_k_space;


  Input>>s_vars.run_dir_;

  Input>>device_choice;
  //Reading device variables
  Input>>graphene_vars.W_,  Input>>graphene_vars.LE_,  Input>>graphene_vars.C_, Input>>graphene_vars.theta_, Input>>graphene_vars.d_min_;
  Input>>graphene_vars.dis_str_, Input>>graphene_vars.dis_seed_;
  Input>>s_vars.cap_choice_;

  //Reading simulation parameters
  Input>>s_vars.base_choice_;
  Input>>s_vars.kernel_choice_;
  Input>>s_vars.M_, Input>>s_vars.R_, Input>>s_vars.dis_real_, Input>>s_vars.seed_, Input>>s_vars.edge_,
  Input>>s_vars.num_parts_,   Input>>cfg.nump_factor;
  Input>>s_vars.a_, Input>>s_vars.b_;

  Input>>s_vars.E_start_,   Input>>s_vars.E_end_;
  Input>>s_vars.eta_;
  Input>>s_vars.E_min_;
  Input>>s_vars.filename_;

  Input>>cfg.sim_type;
  Input>>sim_equation;

  if(sim_equation == "GREENWOOD")
    s_vars.sim_equation_ = KUBO_GREENWOOD;
  if(sim_equation == "BASTIN")
    s_vars.sim_equation_ = KUBO_BASTIN;
  if(sim_equation == "SEA")
    s_vars.sim_equation_ = KUBO_SEA;

  Input>>RAM_size;

  Input>>graphene_vars.filename_;

  Input>>string_is_k_space;
  m_vars.k_space_ = ( string_is_k_space == "imag" );

  Input>>m_vars.range_;
  Input>>m_vars.stgr_str_;
  Input>>m_vars.m_str_;
  Input>>m_vars.rashba_str_;
  Input>>m_vars.KM_str_;
  Input>>m_vars.HLD_str_;

  Input>>s_vars.vel_dir_1_;
  Input>>s_vars.vel_dir_2_;

  Input>>graphene_vars.Bz_;
  Input>>graphene_vars.projector_option_;

  if( !Input )
    throw input_error("legacy input file '" + path + "' ended early or has a value of the wrong type");

  if( cfg.sim_type == "filtered" ){
    Input>>cfg.L_factor,   Input>>f_vars.decRate_;
    Input>>f_vars.energy_center_, Input>>cfg.cutoff_factor;
    Input>>f_vars.att_;

    if( !Input )
      throw input_error("legacy input file '" + path + "': could not read the filter parameters");
  }

  cfg.device_type = device_name(device_choice);

  return cfg;
}
