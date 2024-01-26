#include "solver_vars.hpp"
#include<iostream>

void Station(int millisec, std::string msg ){
  int sec=millisec/1000;
  int min=sec/60;
  int reSec=sec%60;
  std::cout<<msg;
  std::cout<<min<<" min, "<<reSec<<" secs;"<<" ("<< millisec<<"ms) "<<std::endl;
}
