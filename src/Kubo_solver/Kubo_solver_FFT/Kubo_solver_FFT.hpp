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

template<typename T>
class State{
private:
  int D_;
  T* data_;

public:
  ~State(){delete []data_;};

  State(int D) : D_(D){
    data_ = new T[D_];
  };

  State(int D, T data) : D_(D){
    data_ = new T[D_];

#pragma omp parallel for
    for(int i = 0; i < D; i++ )
      data_[i] = data(i);
    
  };

  
  State(State& other_state):D_(other_state.D()){
    data_ = new T[D_];

#pragma omp parallel for
    for(int i = 0; i < D; i++ )
      data_[i] = other_state(i);
    
  };

  
  
  int D(){ return D; };
  T* data(){ return data_; };
  T& operator() (int i){ data_[ i ]; };

  
  State<T>& operator=(const State<T>& other_state){
        if (this == &other_state) 
            return *this; // Handle self-assignment
        

        // Delete existing data if necessary
        delete[] data_;

        D_ = other_state.D();
        data_ = new T[D_];

#pragma omp parallel for
        for (int i = 0; i < D_; i++) 
            data_[i] = other_state(i);
        

        return *this;
  };
  
};


class Kubo_solver_FFT{
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
  type *vec_,
       *p_vec_,
       *pp_vec_,
       *rand_vec_,
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
  void reset_data ( std::vector<T>){ std::fill(r_data_.begin(), r_data_.end(), 0); };
  
  void update_data ( std::vector<type>&, const std::vector<type>&, int );  


  //Heavy duty
  void compute();
  void polynomial_cycle ( type**, int, bool);

  
  void Greenwood_FFTs( type**, type**,  std::vector<type>&, int);
  void Bastin_FFTs ( type**, type**, std::vector<type>&, int);



   
};




class Kubo_solver_FFT_postProcess{//will interpret data_set of points k=0,...,nump-1 as associated to  the energies e_k=a_*cos(MPI*(2*k+0.5))-b_; 
        typedef double  r_value_t;
        typedef std::complex< r_value_t > value_t;

        private:
          Kubo_solver_FFT parent_solver_;
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

