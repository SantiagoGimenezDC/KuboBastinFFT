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



/*


template <typename T>
class State{
  typedef int indexType;
private:
  indexType D_;
  std::vector<T> state_;
  //whats being passed as reference/ as value?
  //operator overloading hmmm
public:
  State(indexType D):D_(D){ state_.reserve(D); };
  State(State& new_state): D_( new_state.dimension() ), state_( new_state.data() ){};
  State(std::vector<T> new_data): D_( new_data.size() ), state_( new_data ){};

  indexType dimension(){return D_; };
  std::vector<T>& data() {return state_; }
  indexType memory_size(){return state_.size() * sizeof(type); };

  
  type operator () (indexType i ){ return state_.at(i); };

  type operator [] (indexType i ){ return state_[i]; };

  void operator = (State& new_state){ state_ = new_state.data(); };  

  State operator + (State& other_state){
    std::vector<T> result(state_);

    #pragma omp parallel for
    for(indexType i; i<D_;i++)
      result += other_state[i]; 

      
    return State( result );
  };  

};




template <class StateType>
class States_buffer{
  typedef int indexType;
private:
  indexType D_, M_;
  std::vector<StateType> states_;
    
public:
  States_buffer(indexType D, indexType M):D_(D),M_(M) {
    states_.reserve(M);

    for( indexType m; m < M_; m++ )
      states_(m).reserve(D_);
    
  };
  
  void push_back( StateType& new_state){ states_.push_back(new_state); };

  indexType memory_size(){ return M_ * states_.at(0).memory_size(); };
  std::vector<StateType>& data(){return states_; };
  
  StateType& operator()(indexType m ){ return states_.at(m); };
  StateType& operator[](indexType m ){ return states_[m]; };
  
  //  type operator()(indexType m, int i){ return states_[m][i]; };

};



template<class StateType>
class Chebyshev_states: public States_buffer<StateType>{
  typedef int indexType;
private:
  Device device_;
  indexType head_num_ = 0;

public:
  Chebyshev_states(Device& device ):device_(device), States_buffer<StateType>( device_.parameters().DIM_, 3 ) {};

  indexType update() {
    if( head_num_ == 0 )
      this->(1) = device_.H_ket( this(0) );
    else{
      this->(2) = device_.H_ket( this(1) ) - this(0);

      this->(0) = this->(1);
      this->(1) = this->(2);
    }
    head_num_++;
    return head_num_;
  };

  
  void reset_init_state( StateType& init_state ){ this(0) = init_state; head_num_ = 0; };


};


class FFT_Operations{
private:
public:
  void preFactors();
  void energy_dependent_ops();
  void compute (States_buffer, States_buffer, std::vector<r_type>, int);

};

*/


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
  r_type *E_points_,
         *conv_R_;
  type *r_data_,
       *final_data_ ;
//--------------------------------------------//

  
public:
  Kubo_solver_FFT();
  Kubo_solver_FFT( solver_vars&, Device&);
  ~Kubo_solver_FFT();
  
  solver_vars& parameters(){ return parameters_; };

  //Initializers
  void allocate_memory();
  void reset_recursion_vectors();
  void reset_Chebyshev_buffers();
  void reset_r_data();

  //Heavy duty
  void compute();

  void polynomial_cycle     ( type**, int);
  void polynomial_cycle_ket ( type**, int);

  void Greenwood_FFTs__imVec ( type**, type**,  type*, int);
  void Bastin_FFTs ( type**, type**, type*, int);



  
  //Post-process
  void Greenwood_postProcess (  int  );
  void Bastin_postProcess ( int  );

  void integration ( r_type*, r_type*, r_type* );
  void partial_integration ( r_type*, r_type*, r_type* );

  void rearrange_crescent_order( r_type* );
  void eta_CAP_correct(r_type*, r_type* );  
  void plot_data   ( std::string, std::string );
  
};


#endif //KUBO_BASTIN_SOLVER_HPP


/*
  
  void Bastin_FFTs__reVec_noEta_2     ( r_type**, r_type**, r_type*, r_type*);
  void Bastin_FFTs__imVec_noEta_2 ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);
  
  void Bastin_FFTs__reVec_noEta     ( r_type**, r_type**, r_type*, r_type*);
  void Bastin_FFTs__imVec_noEta     ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);
  void Bastin_FFTs__imVec_eta       ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);

  void Greenwood_FFTs__reVec_noEta ( r_type**, r_type**, r_type*, r_type*);  
  void Greenwood_FFTs__reVec_eta   ( r_type*, r_type*, r_type*, r_type*);
  
  void Greenwood_FFTs__imVec_eta ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);




*/
