#ifndef CAP_HPP
#define CAP_HPP

#include "static_vars.hpp"


class CAP {
private:
  r_type Emin_, eta_;
public:
  ~CAP();
  CAP(){};
  CAP( r_type Emin,r_type eta) : Emin_(Emin), eta_(eta){};

  void create_CAP(int, int, int,  r_type*);  
};

void create_CAP(int , int , int , r_type , r_type , r_type*);
void eff_contact(int , int , int , r_type , r_type* );

#endif //CAP_HPP
