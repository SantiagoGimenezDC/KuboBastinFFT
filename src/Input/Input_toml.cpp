// Reader for TOML input files. See examples/template/input.toml for a documented example.
//
// Every key that is read is checked for type and range, and any key that is not used by the
// chosen device/solver is reported as an error, so typos and leftover parameters are caught
// before the simulation starts.

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <vector>

#include "toml.hpp"
#include "Input.hpp"


namespace {

std::string join(const std::vector<std::string>& v){
  std::string out;
  for(size_t i = 0; i < v.size(); i++)
    out += ( i ? ", " : "" ) + v[i];
  return out;
}


// A TOML table that remembers which keys were looked up, so leftovers can be reported.
class Section{
private:
  const toml::table* table_;
  std::string path_;               // e.g. "solver.filter"
  std::vector<std::string> known_; // keys looked up, in order (for error messages)
  std::set<std::string> known_set_;

  std::string full(const std::string& key) const { return path_.empty() ? key : path_ + "." + key; }

  void remember(const std::string& key){
    if( known_set_.insert(key).second )
      known_.push_back(key);
  }

  static std::string where(const toml::node& n){
    return " (line " + std::to_string(n.source().begin.line) + ")";
  }

public:
  Section(const toml::table& table, std::string path) : table_(&table), path_(std::move(path)) {}

  bool has(const std::string& key){
    remember(key);
    return table_->contains(key);
  }

  const toml::node& node(const std::string& key){
    remember(key);
    const toml::node* n = table_->get(key);
    if( !n )
      throw input_error("missing required key '" + full(key) + "'");
    return *n;
  }

  // --- typed getters: required ---

  std::string get_string(const std::string& key){
    const toml::node& n = node(key);
    if( !n.is_string() )
      throw input_error("'" + full(key) + "' must be a string" + where(n));
    return *n.value<std::string>();
  }

  r_type get_real(const std::string& key){
    const toml::node& n = node(key);
    if( !n.is_number() )
      throw input_error("'" + full(key) + "' must be a number" + where(n));
    return *n.value<double>();
  }

  int64_t get_int(const std::string& key, int64_t min = INT32_MIN, int64_t max = INT32_MAX){
    const toml::node& n = node(key);
    if( !n.is_integer() )
      throw input_error("'" + full(key) + "' must be an integer" + where(n));
    int64_t v = *n.value<int64_t>();
    if( v < min || v > max )
      throw input_error("'" + full(key) + "' = " + std::to_string(v) + " is out of range ["
			+ std::to_string(min) + ", " + std::to_string(max) + "]" + where(n));
    return v;
  }

  bool get_bool(const std::string& key){
    const toml::node& n = node(key);
    if( !n.is_boolean() )
      throw input_error("'" + full(key) + "' must be true or false" + where(n));
    return *n.value<bool>();
  }

  // One of a fixed set of names; returns its index in `options`.
  int get_choice(const std::string& key, const std::vector<std::string>& options){
    std::string v = get_string(key);
    for(size_t i = 0; i < options.size(); i++)
      if( options[i] == v )
	return int(i);
    throw input_error("'" + full(key) + "' = \"" + v + "\" is not valid; expected one of: " + join(options) + where(node(key)));
  }

  std::vector<int64_t> get_int_array(const std::string& key, size_t size, int64_t min = 0){
    const toml::node& n = node(key);
    const toml::array* arr = n.as_array();
    if( !arr || arr->size() != size )
      throw input_error("'" + full(key) + "' must be an array of " + std::to_string(size) + " integers" + where(n));
    std::vector<int64_t> out;
    for(const toml::node& e : *arr){
      if( !e.is_integer() || *e.value<int64_t>() < min )
	throw input_error("'" + full(key) + "' must be an array of " + std::to_string(size)
			  + " integers >= " + std::to_string(min) + where(n));
      out.push_back(*e.value<int64_t>());
    }
    return out;
  }

  // --- typed getters: optional with default ---

  std::string get_string(const std::string& key, const std::string& def){ return has(key) ? get_string(key) : def; }
  r_type      get_real  (const std::string& key, r_type def)            { return has(key) ? get_real(key)   : def; }
  bool        get_bool  (const std::string& key, bool def)              { return has(key) ? get_bool(key)   : def; }
  int64_t     get_int_or(const std::string& key, int64_t def, int64_t min, int64_t max = INT32_MAX){
    return has(key) ? get_int(key, min, max) : def;
  }
  int get_choice(const std::string& key, const std::vector<std::string>& options, const std::string& def){
    if( has(key) ) return get_choice(key, options);
    for(size_t i = 0; i < options.size(); i++)
      if( options[i] == def ) return int(i);
    return 0;
  }

  // --- sub-tables ---

  Section sub(const std::string& key){
    const toml::node& n = node(key);
    if( !n.is_table() )
      throw input_error("'" + full(key) + "' must be a table" + where(n));
    return Section(*n.as_table(), full(key));
  }

  std::optional<Section> sub_optional(const std::string& key){
    if( !has(key) ) return std::nullopt;
    return sub(key);
  }

  // Throws if the table holds keys that were never looked up.
  void check_unused(const std::string& context = ""){
    for(const auto& [k, n] : *table_){
      std::string key(k.str());
      if( known_set_.count(key) == 0 ){
	std::string msg = "unknown or unused key '" + full(key) + "'" + where(n);
	if( !context.empty() )
	  msg += "; it is not used " + context;
	msg += ". Accepted keys" + std::string( path_.empty() ? " at top level" : " in [" + path_ + "]" )
	  + ": " + join(known_);
	throw input_error(msg);
      }
    }
  }
};



const std::vector<std::string> sim_types  = { "DOS", "normal", "traditional", "filtered" };
const std::vector<std::string> equations  = { "GREENWOOD", "BASTIN", "SEA" };   // order matches enum formula
const std::vector<std::string> caps       = { "Mandelshtam", "Effective_Contact" };   // order matches cap_choice_
const std::vector<std::string> kernels    = { "None", "Jackson" };                    // order matches kernel_choice_
const std::vector<std::string> bases      = { "Direct", "Complex_Phase", "Complex_Phase_real",
					      "FullTrace", "projected_FullTrace" };     // order matches base_choice_

const std::set<std::string> graphene_family = { "Graphene", "ArmchairGraph_RashbaSOC", "Graphene_supercell",
						"SupercellGraph_RashbaSOC", "Graphene_KaneMele", "Graphene_nanoribbon_kspace" };
const std::set<std::string> kane_mele_family = { "Graphene_KaneMele", "Graphene_nanoribbon_kspace" };
const std::set<std::string> rashba_family    = { "ArmchairGraph_RashbaSOC", "SupercellGraph_RashbaSOC" };
const std::set<std::string> file_readers     = { "Read_Hamiltonian", "Read_ConTable", "Read_Siesta", "Read_Siesta_Cropped" };



void read_device(Section device, input_config& cfg){
  device_vars& d = cfg.d_vars;
  model_vars&  m = cfg.m_vars;

  std::string type = device.get_string("type");
  bool known = false;
  for(int i = 0; i <= 10; i++)
    known = known || ( device_name(i) == type );
  if( !known ){
    std::vector<std::string> names;
    for(int i = 0; i <= 10; i++) names.push_back(device_name(i));
    throw input_error("'device.type' = \"" + type + "\" is not valid; expected one of: " + join(names));
  }
  cfg.device_type = type;

  d.W_  = device.get_int("W",  1);
  d.LE_ = device.get_int("LE", 1);
  d.C_  = device.get_int_or("C", 0, 0);

  if( auto dis = device.sub_optional("disorder") ){
    d.dis_str_  = dis->get_real("strength");
    d.dis_seed_ = dis->get_int("seed", 0);
    dis->check_unused();
  }

  // Device-specific parameters: only read (and only accepted) for the devices that use them.
  if( graphene_family.count(type) )
    d.Bz_ = device.get_real("Bz", 0.0);

  if( type == "TBG" ){
    d.theta_ = device.get_real("theta");  // degrees
    d.d_min_ = device.get_real("d_min");
  }

  if( file_readers.count(type) )
    d.filename_ = device.get_string("filename");

  if( kane_mele_family.count(type) ){
    d.projector_option_ = device.get_int_or("projector", 0, 0);
    m.k_space_ = device.get_bool("k_space", false);
    m.range_   = device.get_real("k_range", 1.0);  // used even when k_space = false (build_Hk always runs)
  }

  if( kane_mele_family.count(type) || rashba_family.count(type) ){
    if( auto c = device.sub_optional("couplings") ){
      m.m_str_      = c->get_real("exchange", 0.0);
      m.rashba_str_ = c->get_real("rashba",   0.0);
      if( kane_mele_family.count(type) ){
	m.stgr_str_ = c->get_real("staggered", 0.0);
	m.KM_str_   = c->get_real("kane_mele", 0.0);
	m.HLD_str_  = c->get_real("haldane",   0.0);
      }
      c->check_unused("by device type '" + type + "'");
    }
  }

  device.check_unused("by device type '" + type + "'");
}



void read_solver(Section solver, input_config& cfg){
  solver_vars& s = cfg.s_vars;
  filter_vars& f = cfg.f_vars;

  cfg.sim_type = sim_types[ solver.get_choice("type", sim_types) ];
  bool is_DOS  = ( cfg.sim_type == "DOS" );

  if( !is_DOS || solver.has("equation") )
    s.sim_equation_ = formula( solver.get_choice("equation", equations) );

  s.M_         = solver.get_int("M", 1);
  s.R_         = solver.get_int_or("R", 1, 1);
  s.dis_real_  = solver.get_int_or("disorder_realizations", 1, 1);
  s.seed_      = solver.get_int_or("seed", 0, 0);
  s.num_parts_ = solver.get_int_or("num_parts", 1, 1);

  cfg.nump_factor = solver.get_real("num_p_factor", 1.0);
  if( cfg.nump_factor <= 0 )
    throw input_error("'solver.num_p_factor' must be > 0");

  s.kernel_choice_ = solver.get_choice("kernel", kernels);
  s.base_choice_   = solver.get_choice("base",   bases);
  s.cap_choice_    = solver.get_choice("cap",    caps, "Mandelshtam");
  s.E_min_         = solver.get_real("cap_E_min", 0.0);
  s.eta_           = solver.get_real("eta");

  std::vector<int64_t> vel = solver.get_int_array("velocities", 2);
  s.vel_dir_1_ = vel[0];
  s.vel_dir_2_ = vel[1];


  Section spectrum = solver.sub("spectrum");
  s.edge_ = spectrum.get_real("edge");
  bool has_a = spectrum.has("a"), has_b = spectrum.has("b");
  if( has_a != has_b )
    throw input_error("[solver.spectrum]: give both 'a' and 'b', or neither for automatic bound detection");
  if( has_a ){
    s.a_ = spectrum.get_real("a");
    s.b_ = spectrum.get_real("b");
    if( s.a_ <= 0 )
      throw input_error("'solver.spectrum.a' must be > 0");
  }
  else{
    s.a_ = -1.0;  // solvers detect the spectral bounds when a_ == -1
    s.b_ =  0.0;
  }
  spectrum.check_unused();


  if( cfg.sim_type == "filtered" ){
    Section filter = solver.sub("filter");
    f.decRate_        = filter.get_int("decimation_rate", 1);
    f.energy_center_  = filter.get_real("energy_center");
    cfg.cutoff_factor = filter.get_real("cutoff_factor");
    cfg.L_factor      = filter.get_real("L_factor", 40.0);
    f.att_            = filter.get_real("attenuation", 96.0);
    filter.check_unused();
  }

  solver.check_unused("by solver type '" + cfg.sim_type + "'");
}

} // namespace



input_config read_toml_input(const std::string& path){
  toml::table root;
  try{
    root = toml::parse_file(path);
  }
  catch(const toml::parse_error& err){
    std::ostringstream msg;
    msg << "could not parse '" << path << "': " << err.description()
	<< " (line " << err.source().begin.line << ", column " << err.source().begin.column << ")";
    throw input_error(msg.str());
  }

  input_config cfg;
  Section top(root, "");

  cfg.s_vars.run_dir_  = top.get_string("run_dir", "./");
  cfg.s_vars.filename_ = top.get_string("output");

  read_device(top.sub("device"), cfg);
  read_solver(top.sub("solver"), cfg);

  top.check_unused();

  return cfg;
}
