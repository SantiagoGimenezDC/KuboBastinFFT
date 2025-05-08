#ifndef DEVICE_HPP
#define DEVICE_HPP

#include "../static_vars.hpp"
#include "../CAP.hpp"
#include "../Random.hpp"
#include <iostream>


struct device_vars{
  size_t W_, LE_, C_, DIM_, SUBDIM_,  DIS_DIM_, dis_seed_, projector_option_;
  r_type dis_str_, theta_, d_min_, Bz_;

  r_type DOS_corr_=1.0;
  std::string run_dir_, filename_;
};


class Device{
private:
  device_vars device_vars_;
  Random rng_;

  r_type sysLength_;
  r_type sysSubLength_;

  r_type *dis_, *damp_;
  
public:
  ~Device(){ delete []dis_; };
  Device( device_vars& device_vars ) : device_vars_( device_vars ), rng_( device_vars.dis_seed_ ){

    

    if(this->parameters().dis_str_ !=0 ){

      dis_ = new r_type[ 4*device_vars_.DIM_ ]; // the 4 is a hack to make it work with the RashbaSOC!!
      #pragma omp parallel for
      for(size_t i=0; i<4*device_vars_.DIM_; i++)// the 4 is a hack to make it work with the RashbaSOC!!
        dis_[i] = 0.0;
    }


    if(false){//Turned off for KaneMele simulations
      std::cout<< "Heres the damp op dim:  "<<device_vars_.DIM_<<std::endl;
      damp_ = new r_type[ device_vars_.DIM_ ];
      #pragma omp parallel for
      for(size_t i=0; i<device_vars_.DIM_; i++)
        damp_[i] = 1.0;
    }

  };


  virtual int unit_cell_size(){return 1;};
  void set_sysLength(r_type sysLength){sysLength_=sysLength;};
  r_type sysLength(){return sysLength_;};
  void set_sysSubLength(r_type sysSubLength){sysSubLength_=sysSubLength;};
  r_type sysSubLength(){return sysSubLength_;};

  void set_dis(r_type* new_dis){ dis_=new_dis; };
  void set_damp_op(r_type* new_damp){ damp_ = new_damp; };

  r_type* dis(){ return dis_; };
  r_type* damp_op(){ return damp_; };
  virtual void projector(type*){};
  virtual bool isKspace(){ return false; };
  virtual void Uk_ket(  type*, type*){};
  virtual void to_kSpace(type* , const type*, int ){};
  
  virtual r_type Hamiltonian_size() = 0;  

  virtual void build_Hamiltonian() = 0;
  virtual void damp   ( r_type* ) = 0;
  virtual void update_dis( r_type* ){};
  virtual void update_dis( r_type*, r_type* ){}; 
  virtual void adimensionalize ( r_type,  r_type ) = 0;
  
  virtual void rearrange_initial_vec(type*) ; //very hacky
  virtual void traceover(type*, type*, int, int) ;
  
  Random& rng(){return rng_;};
  //  CAP& cap(){return cap_;};
  device_vars& parameters(){return device_vars_; };


  virtual void update_cheb_filtered ( type*, type*, type*, r_type*, r_type* , type) {};  

  virtual void update_cheb ( type*, type*, type* ){};  
  virtual void update_cheb ( type*, type*, type*, r_type*, r_type* ) = 0;  
  virtual void update_cheb ( type*, type*, type*, r_type*, r_type , r_type ){};
  virtual void update_cheb ( int ,  int, type*, type*, type*, type*, r_type*, r_type , r_type ){};

  virtual void H_ket ( type*, type*) = 0;
  virtual void H_ket ( type*, type*, r_type*, r_type*) = 0;  

  virtual void vel_op (type*, type*){};
  virtual void vel_op (type*, type*, int){};
  virtual void J (type*, type*, int){};

  
  virtual void setup_velOp() = 0;
  
  void Anderson_disorder();
  void Anderson_disorder( r_type* );  

  void minMax_EigenValues( int , r_type& , r_type& );
};



#endif //DEVICE_HPP
