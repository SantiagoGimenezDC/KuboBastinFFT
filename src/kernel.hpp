#ifndef KERNEL_HPP
#define KERNEL_HPP


#include"static_vars.hpp"
#include<cmath>
#include<fstream>




class Kernel {
public:
  virtual ~Kernel(){};
  Kernel(){};

  virtual type term(const int m) = 0;
    
};





class Jackson : public Kernel{
  private:
    int Ml_;
  public:
    virtual ~Jackson(){};
    Jackson(int M) : Ml_(M){};


    virtual type term(const int m){
      return 1/(Ml_+1.0)*
           (
   	     (Ml_-m+1.0)*
             cos(M_PI*m / (Ml_+1.0)) +
             sin(M_PI*m / (Ml_+1.0)) /
             tan(M_PI   / (Ml_+1.0))
	   );
    };
    
};



class None : public Kernel{
  public:
    virtual ~None(){};
    None(){};

    virtual type term(const int ){
      return 1.0;
    };
    
};










inline type kernel(const int m){
  return 1/(M_+1.0)*
         (
 	   (M_-m+1.0)*
           cos(M_PI*m / (M_+1.0)) +
           sin(M_PI*m / (M_+1.0)) /
           tan(M_PI   / (M_+1.0))
	 );
}

#endif //KERNEL_HPP
