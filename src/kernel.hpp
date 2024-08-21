#ifndef KERNEL_HPP
#define KERNEL_HPP


#include"static_vars.hpp"
#include<cmath>
#include<fstream>




class Kernel {
public:
  virtual ~Kernel(){}
  Kernel(){};
  virtual r_type term(const int m, const int M) = 0;
    
};



class Lorentz : public Kernel{
private:
  const r_type lambda_ = 3.5;
  public:
    Lorentz(){};
    virtual r_type term(const int m, const int M){
      return sinh(lambda_ * ( 1.0 - r_type(m) / r_type(M) ) ) / sinh( lambda_ );
    };
    
};



class Jackson : public Kernel{
  public:
    Jackson(){};


    virtual r_type term(const int m, const int M){
      return 1.0/(M+1.0) *
           (
   	     (M-m+1.0) *
             cos( M_PI * m / (M + 1.0)) +
             sin( M_PI * m / (M + 1.0)) /
             tan( M_PI     / (M + 1.0))
	   );
    };
    
};



class None : public Kernel{
  public:
    virtual ~None(){};
    None(){};

  virtual r_type term(const int, const int ){
      return 1.0;
    };
    
};


#endif //KERNEL_HPP
