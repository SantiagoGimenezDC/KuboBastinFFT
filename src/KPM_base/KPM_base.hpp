#ifndef KPM_BASE_HPP
#define KPM_BASE_HPP

#include<string>
#include<iostream>
#include<eigen-3.4.0/Eigen/Core>

#include"../static_vars.hpp"
#include "../Device/Device.hpp"
#include "../Device/Graphene.hpp"
#include "../Device/ArmchairGraph_RashbaSOC.hpp"

#include "../kernel.hpp"
#include "../vec_base.hpp"

#include "../Kubo_solver/solver_vars.hpp"

#include "../States.hpp"


class KPM_base{
private:
  solver_vars parameters_;  
  
  Device&  device_;
  Kernel*   kernel_;
  CAP*      cap_;
  Vec_Base* vec_base_;

  r_type* dmp_op_;
  type* rand_vec_;
public:
  KPM_base();
  KPM_base( solver_vars&, Device&);
  virtual ~KPM_base();

  Device& device(){ return device_; };
  solver_vars& parameters(){ return parameters_; };
  Kernel& kernel(){return *kernel_;};
  CAP& cap(){return *cap_;};
  Vec_Base& vec_base(){return *vec_base_;};

  type* rand_vec(){return rand_vec_;};
  r_type* dmp_op(){return dmp_op_;};
  
  //Initializers
  virtual void allocate_memory() = 0;
  virtual void initialize_device();

  //Heavy duty
  void compute();
  virtual void compute_rand_vec( int) = 0;
};





class DOS_output{//will interpret data_set of points k=0,...,nump-1 as associated to  the energies e_k=a_*cos(MPI*(2*k+0.5))-b_; 
  typedef double  r_value_t;
  typedef std::complex< r_value_t > value_t;

  private:
    device_vars& device_parameters_;
    solver_vars& parameters_;
    Kernel*   kernel_;

    std::vector<r_type>
      E_points_,
      r_data_,
      partial_result_,
      conv_R_max_,
      conv_R_av_;

  public:       
    DOS_output(device_vars&, solver_vars& );
    ~DOS_output(){};

  
    solver_vars& parameters(){return parameters_;};
    device_vars& device_parameters(){return device_parameters_;};
  
    std::vector<r_type>& partial_result(){return partial_result_;};
    std::vector<r_type>& r_data(){return r_data_;};

    void update_data(std::vector<type>&, std::vector<type>&, int);
  
    void operator()( const std::vector<type>&, const std::vector<type>&, int);
    void plot_data   ( const std::string&, const std::string& );

};



class KPM_DOS_solver: public KPM_base{
private:
   DOS_output output_;
   std::vector<type> moments_r_, moments_acc_;
  

public:
  KPM_DOS_solver();
  KPM_DOS_solver( solver_vars& parameters, Device& device):KPM_base(parameters, device), output_(device.parameters(), parameters){
    moments_r_   = std::vector<type>(parameters.M_, 0.0);
    moments_acc_ = std::vector<type>(parameters.M_, 0.0);
  };

  
  virtual ~KPM_DOS_solver(){/*delete cheb_vectors_;*/};

  
  //Initializers
  virtual void allocate_memory();

  //Heavy duty
  virtual void compute_rand_vec( int);

     
};





#endif //KPM_BASE_HPP

