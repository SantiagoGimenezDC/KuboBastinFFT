#ifndef TIME_STATION_2_HPP
#define TIME_STATION_2_HPP


#include<chrono>
#include<iostream>
#include"../static_vars.hpp"


class time_station_2{
private:
  std::chrono::steady_clock::time_point start_, end_;
  long int time_microSec_ = 0;
public:
  time_station_2(){};

  time_station_2(int time, std::string msg){
    print_msg(time, msg);
  };//for whenever all has been added previously


  inline long int time(){ return time_microSec_; };
  
  inline void operator += (time_station_2& other){ time_microSec_ += other.time();};
  
  inline void start(){ start_ = std::chrono::steady_clock::now();}


  inline void start(std::string msg ){
    start_ = std::chrono::steady_clock::now();
    print_msg(time_microSec_, msg);
  };


  
  inline long int stop(){
    end_ = std::chrono::steady_clock::now();
    time_microSec_ += std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count() ;
    return time_microSec_;
  }


  
  inline long int stop(std::string msg ){
    
    end_ = std::chrono::steady_clock::now();
    time_microSec_ +=  std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count() ;

    print_msg(time_microSec_, msg);

    return time_microSec_;
  };


  
    
  
  inline void print_msg(int time, std::string msg ){
    
    int sec   =  time/1000000;
    int min   = sec / 60;
    int hour  = min / 60;
    int reSec = sec % 60;
    int reMin = min % 60;
  
    std::cout<<msg;

    if(hour>0)
      std::cout<<hour<<" hrs, "<<reMin<<" mins;"<<std::endl;
    else if(min>0)
      std::cout<<min<<" mins, "<<reSec<<" secs;"<<std::endl;
    else if(sec>0)
      std::cout<<sec<<" secs;"<<" ("<< r_type(time)/r_type(1000)<<"ms) "<<std::endl;
    else
      std::cout<< r_type(time)/r_type(1000)<<" ms "<<std::endl;    
 }



  
  inline void stop_add_add(int timeover, int* time, std::string msg ){//specific weird case I need

    end_ = std::chrono::steady_clock::now();
    time_microSec_ =  std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count() ;

    *time += time_microSec_;

    print_msg(time_microSec_+timeover, msg);
  };


  
};

#endif //TIME_STATION_2_HPP
