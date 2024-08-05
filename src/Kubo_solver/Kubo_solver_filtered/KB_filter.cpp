#include<cmath>
#include<fstream>
#include<iostream>

#include<boost/math/special_functions/bessel.hpp> 

#include "KB_filter.hpp"
#include "../../kernel.hpp"

KB_filter::KB_filter(filter_vars& parameters): parameters_(parameters){
  
  int M_ext   = parameters_.M_ext_,
      M       = parameters_.M_,
      decRate = parameters_.decRate_,
      L       = this->parameters().L_,
      Np      = (L-1)/2;



  if( M_ext % decRate != 0 ){
    std::cout<<"M_ext is being increased by "<< ( decRate - M_ext % decRate )<<" to match divisibility by the decimation rate" <<std::endl;

    M_ext              += ( decRate - M_ext % decRate ) ;
    parameters_.M_ext_ = M_ext;
    parameters_.f_cutoff_ = 0.9 * parameters_.M_ext_/ ( 2 * parameters_.decRate_ ); //a default estimate of the cutoff. Verify.

  }
  
  if(parameters_.L_%2 == 0)
    std::cout<<"Filter length L should be odd."<<std::endl;

  if(parameters_.L_ > M)
    std::cout<<"Filter length L should smaller than M."<<std::endl;

  

    
  int nump = 0;

  for( int m = 0; m < M_ext; m++)
    if( m % decRate == 0 ){
      nump++;
      if( ( m < M + Np ) ||  ( m > M_ext - 1 - Np ) )
        decimated_list_.push_back( m );
    }

  //  for( int m = 0; m < decimated_list_.size(); m++)
  //  std::cout<<decimated_list_[m]<<"  "<<M_ext - 1 - Np<<" "<<decimated_list_.size()<<std::endl;
  
  
  
  M_dec_ = decimated_list_.size();
  parameters_.nump_ = nump;

  /*  M_dec_ = M / decRate ;
  if( (M - 1) % decRate == 0 && (decRate != 1) ){
    M_dec_++;
    parameters_.nump_=M_dec_;
    }*/


  //nump = M_dec_;

  
  r_type att = parameters.att_;    

  if(att<21)
    beta_ = 0;
  else if( att >= 21 && att < 50)
    beta_ = 0.5842*pow(( att - 21.0),0.4)+0.07886 * ( att - 21.0);
  else 
    beta_ = 0.1102 * ( att - 8.7);

}
  

void KB_filter::compute_filter(){   
  int L = this->parameters().L_,
    M_ext = parameters_.M_ext_;
  
  int Np = (L-1)/2;
 
  r_type  f_a = 0,
    f_s = M_ext,
    f_b = this->parameters().f_cutoff_;


  Eigen::VectorXd A(Np+1);

  A(0) = 2 * ( f_b - f_a ) / f_s;
  
  for(int l = 1; l <= Np; l++)
    A(l) = ( sin( 2 * M_PI * l * f_b / f_s ) - sin( 2 * M_PI * l * f_a / f_s ) ) / ( M_PI * l );



  KB_window_.resize(L);
  KB_window_.setZero();
  for(int l = 0; l <= Np; l++){
    KB_window_(Np+l) = A(l)*boost::math::cyl_bessel_i(0,  beta_ * sqrt (1.0 - pow(  double(l)/double(Np) , 2.0) ) ) / boost::math::cyl_bessel_i(0, beta_);
    KB_window_(Np-l) = KB_window_(l+Np);
  }


  

  parameters_.L_eff_=L;
  double thresh = 1E-4;
  for(int i=3; i<Np;i++){
    if(abs(KB_window_(Np+i)) / KB_window_(Np)<thresh && abs(KB_window_(Np+i-1)) /KB_window_(Np)<thresh && abs(KB_window_(Np+i-2))/KB_window_(Np)<thresh && abs(KB_window_(Np+i-3))/KB_window_(Np)<thresh ){
      parameters_.L_eff_=i;
      break;
    }
  }
  
  std::cout<<" Effective filter length: "<<parameters_.L_eff_<<std::endl;
  
  if(Np == 0 )
    KB_window_(0) = 1.0;// for testing

}

/*
void KB_filter::print_filter(std::string runDir){  

  Eigen::VectorXdc w(L), f_w(L), phase;  
  std::ofstream dataP;
  dataP.open(runDir+"KaiserBesselWindow.txt");
  dataP<<KB_window_;
  dataP.close();


        
   std::string exestring=
         "gnuplot<<EOF                                               \n"
         "set encoding utf8                                          \n"
         "set terminal pngcairo enhanced                             \n"

         "unset key  \n"

         "set output '"+runDir+"KaiserBesselWindow.png'                \n"

        "plot '"+runDir+"KaiserBesselWindow.txt' w p ls 7 ps 0.25 lc 2;  \n"
         "EOF";
     
   char exeChar[exestring.size() + 1];
   strcpy(exeChar, exestring.c_str());    
   system(exeChar);



  w=KBwindow;///w.norm();      
   f_w=w;

  quick_CFT(f_w);

  precision ref= abs(f_w(L/2-1));


  MatrixXp f_w_fullData(L,2);
  f_w_fullData.col(1)=f_w.real();

  for(int n=0; n<L;n++){
    f_w_fullData(n,0) = n-L/2;
     f_w_fullData(n,1) = log(  abs(f_w(n)/ref)*abs(f_w(n)/ref)  ) ;
  }

  
  std::ofstream dataP2;
  dataP2.open(runDir+"KaiserBesselSpectrum.txt");
  dataP2<<f_w_fullData;
  dataP2.close();


        
   std::string exestring2=
         "gnuplot<<EOF                                               \n"
         "set encoding utf8                                          \n"
         "set terminal pngcairo enhanced                             \n"

         "unset key  \n"
     //     "set xrange ["+std::to_string(-L/2)+":0]\n" 
         


         "set output '"+runDir+"KaiserBesselSpectrum.png'                \n"

        "plot '"+runDir+"KaiserBesselSpectrum.txt' w p ls 7 ps 0.25 lc 2;  \n"
         "EOF";
     
   char exeChar2[exestring2.size() + 1];
   strcpy(exeChar2, exestring2.c_str());    
   system(exeChar2);



   
  for(int l=0;l<L;l++)
    f_w_fullData(l,1)=atan(f_w(l).imag()/f_w(l).real());

  KBphase=f_w_fullData.col(1);
  
  std::ofstream dataP3;
  dataP3.open(runDir+"KaiserBesselPhase.txt");
  dataP3<<f_w_fullData;
  dataP3.close();


        
   std::string exestring3=
         "gnuplot<<EOF                                               \n"
         "set encoding utf8                                          \n"
         "set terminal pngcairo enhanced                             \n"

         "unset key  \n"
     //     "set xrange ["+std::to_string(-L/2)+":0]\n" 
         


         "set output '"+runDir+"KaiserBesselPhase.png'                \n"

        "plot '"+runDir+"KaiserBesselPhase.txt' w p ls 7 ps 0.25 lc 2;  \n"
         "EOF";
     
   char exeChar3[exestring3.size() + 1];
   strcpy(exeChar3, exestring3.c_str());    
   system(exeChar3);

  
}
*/

void KB_filter::post_process_filter(type**  polys, int subDim){

  std::complex<r_type> ImUnit(0,1.0);
  
  int M   = this->parameters().M_,
    M_ext = this->parameters().M_ext_,
    L     = this->parameters().L_,
    Np    = (L-1)/2,
    decRate = this->parameters().decRate_,
    k_dis = this->parameters().k_dis_;  


  //  Jackson kernel;

  Eigen::Matrix<std::complex<r_type>,-1,-1> filtered_polys(subDim,M);

  bool cyclic = true;

    
  for(int m=0; m<M;m++){
    std::complex<r_type> phase = ( 2 - ( m == 0) ) * std::polar(1.0,  M_PI * m * (  - 2 * k_dis + 0.5) / M_ext ) ;//( std::polar( 1.0, (  2* M_PI * (r_type) ( - m * k_dis - 0.) / (r_type) M ) ) );
    #pragma omp parallel for
    for(int l=0;l<subDim;l++)
      polys[m][l] *= phase;
  }

  
  filtered_polys.setZero();
  
  for(int m=0;m<M;m++)
    if( m % decRate == 0 ){
      if(cyclic){
	
        for(int i=-Np;  i<=Np; i++){	
          if(m+i<0){
#pragma omp parallel for
	    for(int l=0;l<subDim;l++)
  	      filtered_polys(l,m) += polys[M+m+i][l] * KB_window_(Np+i);
	    //std::cout<<"Filt: "<< M+m+i<<"  "<<m<<" "<<std::abs(i)<<std::endl;
	  }      
          else if(m+i>=M){
#pragma omp parallel for
            for(int l=0;l<subDim;l++)
	      filtered_polys(l,m) += polys[m+i-M][l] * KB_window_(Np+i);
	    //std::cout<<"Filt: "<< m+i-M<<"  "<<m<<" "<<std::abs(i)<<std::endl;
          }      
          else{
#pragma omp parallel for
            for(int l=0;l<subDim;l++)
 	      filtered_polys(l,m) += polys[m+i][l] * KB_window_(Np+i);  
          }      
	}
      }
      else{
	
        for(int i=-Np;  i<=Np; i++){
          if(m+i>=0 && m+i<M){
#pragma omp parallel for
	    for(int l=0;l<subDim;l++)
	      filtered_polys(l,m)+=polys[m+i][l] * KB_window_(Np+i);  
	  }
        }
      }
    }
  
  for(int m=0, i=0; i<=M-1; m++, i+=decRate){
#pragma omp parallel for
     for(int l=0;l<subDim;l++)
       polys[m][l] = filtered_polys(l,i);
       }
}


