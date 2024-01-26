#ifndef KB_FILTER_HPP
#define KB_FILTER_HPP


#include "../static_vars.hpp"
#include "../complex_op.hpp"

#include<string>
#include <eigen-3.4.0/Eigen/Core>

struct filter_vars{
  bool post_filter_, filter_;
  int M_, L_, k_dis_, decRate_, nump_;
  r_type f_cutoff_, att_;
};


class KB_filter{
private:
  filter_vars& parameters_;
  r_type beta_;
  Eigen::VectorXd E_points_, KB_window_;
  
public:
  KB_filter(filter_vars&);
  filter_vars parameters(){return parameters_;};
  void compute_filter();
  void print_filter(std::string);
  r_type* E_points(){return E_points_.data();};
  void post_process_filter(type**  , int );
};
#endif //KB_FILTER_HPP
