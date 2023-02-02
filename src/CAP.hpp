#ifndef CAP_HPP
#define CAP_HPP

#include "static_vars.hpp"


class CAP {
private:
  type Emin_, eta_;
public:
  ~CAP();
  CAP(){};
  CAP( type Emin,type eta) : Emin_(Emin), eta_(eta){};

  void create_CAP(int, int, int,  type*);  
};

void create_CAP(int , int , int , type , type , type*);

#endif //CAP_HPP
