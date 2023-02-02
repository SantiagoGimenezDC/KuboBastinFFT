#ifndef KERNEL_HPP
#define KERNEL_HPP


#include"static_vars.hpp"
#include<cmath>
#include<fstream>




class Kernel {
public:
  virtual ~Kernel(){}
  Kernel(){};

  virtual type term(const int m, const int M) = 0;
    
};





class Jackson : public Kernel{
  public:
    Jackson(){};


  virtual type term(const int m, const int M){
      return 1/(M+1.0) *
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

  virtual type term(const int, const int ){
      return 1.0;
    };
    
};


#endif //KERNEL_HPP
