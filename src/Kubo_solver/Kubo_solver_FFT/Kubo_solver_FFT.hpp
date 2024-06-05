#ifndef KUBO_BASTIN_SOLVER_HPP
#define KUBO_BASTIN_SOLVER_HPP

#include<string>
#include<iostream>
#include<eigen-3.4.0/Eigen/Core>

#include"../../static_vars.hpp"
#include "../../Device/Device.hpp"
#include "../../Device/Graphene.hpp"
#include "../../kernel.hpp"
#include "../../vec_base.hpp"

#include "../solver_vars.hpp"

#include "States.hpp"

template<class State_T>
class Chebyshev_states: public States_buffer<State_T>{
private:
  Device& device_;
  int head_num_ = 0;

public:
  Chebyshev_states(Device& device ):device_(device), States_buffer<State_T>( device.parameters().DIM_, 3 ) {};

  int update() {
    
    if( head_num_ == 0 )
      device_.H_ket( (*this)(1).data(), (*this)(0).data() );

    else
      device_.update_cheb( (*this)(2).data(), (*this)(1).data(), (*this)(0).data() );

      
    head_num_++;
    return head_num_;
  };

  State_T& head(){ return (*this)(2); };
  
  void reset( State_T& init_state ){
    (*this)(0) = init_state;
    head_num_ = 0;
  };

};



class Kubo_solver_FFT{
  //typedef  type** storageType;
  typedef  States_buffer_sliced< State<type> >& storageType;
private:
  solver_vars parameters_;
  formula sym_formula_;
  
  
  Device&  device_;
  Kernel*   kernel_;
  CAP*      cap_;
  Vec_Base* vec_base_;

 


  
//---------------Large vectors----------------//  
  type **bras_ ,
       **kets_ ;
  
  //Recursion Vectors
  type *rand_vec_,
       *tmp_;
  
  //Auxiliary - disorder and CAP vectors
  r_type *dmp_op_,
         *dis_vec_ ;
//--------------------------------------------//


  
  
//---------------Dataset vectors--------------//
  std::vector<type> r_data_,
                    final_data_ ;
//--------------------------------------------//

  
public:
  Kubo_solver_FFT();
  Kubo_solver_FFT( solver_vars&, Device&);
  ~Kubo_solver_FFT();

  Device& device(){ return device_; };
  solver_vars& parameters(){ return parameters_; };
  formula simulation_formula(){ return sym_formula_; };


  //Initializers
  void allocate_memory();
  void initialize_device();
  void reset_recursion_vectors();
  void reset_Chebyshev_buffers();

  template<typename T>
  inline
  void reset_data ( std::vector<T>& data ){ std::fill(data.begin(), data.end(), 0.0); };
  
  void update_data ( std::vector<type>&, const std::vector<type>&, int  );



  //Heavy duty
  void compute();
  void polynomial_cycle ( storageType, Chebyshev_states< State<type> >&, int, bool);

  
  void Greenwood_FFTs( storageType, storageType, std::vector<type>&, int);
  void Bastin_FFTs   ( storageType, storageType, std::vector<type>&, int);



   
};




class Kubo_solver_FFT_postProcess{//will interpret data_set of points k=0,...,nump-1 as associated to  the energies e_k=a_*cos(MPI*(2*k+0.5))-b_; 
        typedef double  r_value_t;
        typedef std::complex< r_value_t > value_t;

        private:
          Kubo_solver_FFT& parent_solver_;
          std::vector<r_type>
	    E_points_,
	    prev_partial_result_,
	    conv_R_max_,
	    conv_R_av_;

  
        public:
       
          Kubo_solver_FFT_postProcess( Kubo_solver_FFT& );
          ~Kubo_solver_FFT_postProcess(){};

  
          void operator()( const std::vector<type>&, const std::vector<type>&, int);
  
          void Greenwood_postProcess ( const std::vector<type>&, const std::vector<type>&, int );
          void Bastin_postProcess    ( const std::vector<type>&, const std::vector<type>&, int);

          void integration         ( const std::vector<r_type>&, const std::vector<r_type>&, std::vector<r_type>& );
          void integration_linqt   ( const std::vector<r_type>&, const std::vector<r_type>&, std::vector<r_type>& );
          void partial_integration ( const std::vector<r_type>&, const std::vector<r_type>&, std::vector<r_type>& );

  
          void rearrange_crescent_order( std::vector<r_type>& );
  
          void eta_CAP_correct(std::vector<r_type>&, std::vector<type>& );
  
          void plot_data   ( const std::string&, const std::string& );


    };



#endif

