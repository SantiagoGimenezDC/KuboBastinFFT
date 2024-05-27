#ifndef KB_FILTER_HPP
#define KB_FILTER_HPP


#include "../static_vars.hpp"
#include "../complex_op.hpp"

#include<string>
#include <eigen-3.4.0/Eigen/Core>

struct filter_vars{
  bool post_filter_, filter_;
  int M_, M_ext_, L_, L_eff_, k_dis_, decRate_, nump_;
  r_type f_cutoff_, att_, energy_center_;
};


class KB_filter{
private:
  filter_vars& parameters_;
  r_type beta_;
  Eigen::VectorXd E_points_, KB_window_;
  int M_dec_;
  
public:
  KB_filter(filter_vars&);
  filter_vars parameters(){return parameters_;};
  void compute_filter();
  void print_filter(std::string);
  r_type* E_points(){return E_points_.data();};
  void post_process_filter(type**  , int );

  void compute_k_dis(r_type a, r_type b){
    r_type adim_e_center = (parameters_.energy_center_ + b )/a;
    parameters_.k_dis_ =  int ( (  r_type(parameters_.M_ext_) * ( std::acos ( adim_e_center )  ) / ( 2.0 * M_PI ) - 0.25  ) );

    int k_dis = parameters_.k_dis_,
        M_ext = parameters_.M_ext_,
        nump = parameters_.nump_;

    if( nump % 2 == 1 ){
       for(int k = 0; k < nump/2 ; k++){
	 E_points_( k )                = cos( 2 * M_PI * ( k - k_dis + 0.25 ) / M_ext );             
	 E_points_( nump / 2 + k + 1 ) = cos( 2 * M_PI * ( nump / 2 - ( k - k_dis + 0.25 ) )  / M_ext );
       }
       E_points_( nump / 2 )           = cos( 2 * M_PI * ( nump / 2 - ( - k_dis + 0.25 ) )  / M_ext );             
    }
    else
      for(int k = 0; k < nump/2 ; k++){
	E_points_( k )            = cos( 2 * M_PI * ( k - k_dis + 0.25 ) / M_ext );             
	E_points_( nump / 2 + k ) = cos( 2 * M_PI * ( nump / 2 - ( k - k_dis + 0.25 ) )  / M_ext );
      }


  };
  
  int M_dec(){return M_dec_;};
  r_type* KB_window(){return KB_window_.data();};
};
#endif //KB_FILTER_HPP
