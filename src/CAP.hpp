#ifndef CAP_HPP
#define CAP_HPP

#include "static_vars.hpp"


class CAP {
private:
  r_type Emin_, eta_;
public:
  virtual ~CAP(){};
  CAP( r_type Emin,r_type eta) : Emin_(Emin), eta_(eta){};

  r_type Emin(){return Emin_;};
  r_type eta(){return eta_;};  

  virtual void create_CAP(int, int, int,  r_type*) = 0;  
};



class Mandelshtam : public CAP {
public:
  virtual ~Mandelshtam(){};
  Mandelshtam( r_type Emin,r_type eta) : CAP(Emin, eta){};

  virtual void create_CAP(int, int, int,  r_type*);  
};



class Effective_Contact : public CAP {
public:
  virtual ~Effective_Contact(){};
  Effective_Contact( r_type Emin,r_type eta) : CAP(Emin, eta){};

  virtual void create_CAP(int, int, int,  r_type*);  
};



void create_CAP(int , int , int , r_type , r_type , r_type*);
void eff_contact(int , int , int , r_type , r_type* );

#endif //CAP_HPP
