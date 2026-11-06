#ifndef INPUT_HPP
#define INPUT_HPP

#include <memory>
#include <string>
#include <stdexcept>

#include "../static_vars.hpp"
#include "../Device/Device.hpp"
#include "../Kubo_solver/solver_vars.hpp"
#include "../Kubo_solver/Kubo_solver_filtered/KB_filter.hpp"


// Parameters that are only passed to the constructors of some devices
// (Graphene_KaneMele, Graphene_nanoribbon_kspace and the Rashba ribbons).
struct model_vars{
  bool   k_space_    = false;
  r_type range_      = 0.0;

  r_type stgr_str_   = 0.0;  // staggered potential
  r_type m_str_      = 0.0;  // exchange
  r_type rashba_str_ = 0.0;
  r_type KM_str_     = 0.0;  // Kane-Mele
  r_type HLD_str_    = 0.0;  // Haldane
};


// Everything main() needs to set up and run a simulation.
struct input_config{
  std::string sim_type;     // DOS | normal | traditional | filtered
  std::string device_type;  // class name of the device, e.g. "Graphene_KaneMele"

  solver_vars s_vars{};
  device_vars d_vars{};
  filter_vars f_vars{};
  model_vars  m_vars{};

  // Raw input factors, turned into derived quantities by finalize_input().
  r_type nump_factor   = 1.0;
  r_type L_factor      = 40.0;
  r_type cutoff_factor = 1.0;
};


class input_error : public std::runtime_error{
public:
  using std::runtime_error::runtime_error;
};


// Reads a .toml input file, or a legacy positional input file for any other extension.
// Throws input_error on invalid input.
input_config read_input(const std::string& path);

input_config read_toml_input  (const std::string& path);
input_config read_legacy_input(const std::string& path);

// Computes derived quantities (DIM_, num_p_, filter lengths, ...) shared by both readers.
void finalize_input(input_config&);

// Device names accepted in the input, indexed by the legacy integer device_choice.
const std::string& device_name(int legacy_choice);

// Builds the device selected in the input; the caller owns it. Throws input_error for an unknown device type.
std::unique_ptr<Device> make_device(input_config&);

#endif //INPUT_HPP
