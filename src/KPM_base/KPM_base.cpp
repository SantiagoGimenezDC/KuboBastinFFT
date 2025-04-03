#include<iostream>
#include<fstream>
#include<string>
#include<cmath>
#include<chrono>
#include<ctime>
#include<omp.h>
#include<iomanip>
#include<cstdlib>
#include<thread>
#include<complex>
#include<cstring>

#include<fftw3.h>

 

#include "../complex_op.hpp"
#include "KPM_base.hpp"

#include "../Kubo_solver/time_station_2.hpp"

#include <sys/stat.h>
#include <sys/types.h>

#include <filesystem>
#include "../Device/Graphene_KaneMele.hpp"


namespace fs = std::filesystem;



void KPM_base::initialize_device(){

  device_.build_Hamiltonian();
  device_.setup_velOp();
  
  if(parameters_.a_ == 1.0){
    r_type Emin, Emax;
    device_.minMax_EigenValues(300, Emax,Emin);

    
    parameters_.a_ =  ( Emax - Emin ) / ( 2.0 - parameters_.edge_ );
    parameters_.b_ = -( Emax + Emin ) / 2.0;

  }
  
  device_.adimensionalize( parameters_.a_, parameters_.b_ );
}



KPM_base::KPM_base(solver_vars& parameters, Device& device) : parameters_(parameters), device_(device)
{

  if(parameters_.cap_choice_ == 0)
    cap_      = new Mandelshtam(parameters_.E_min_, parameters_.eta_/parameters_.a_);
  else if(parameters_.cap_choice_==1)
    cap_      = new Effective_Contact(parameters_.E_min_, parameters_.eta_/parameters_.a_);

  
  if(parameters_.base_choice_ == 0)
    vec_base_ = new Direct(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_ == 1 )
    vec_base_ = new Complex_Phase(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_ == 2 )
    vec_base_ = new Complex_Phase_real(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_ == 3 )
    vec_base_ = new FullTrace(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_ == 4 )
    vec_base_ = new projected_FullTrace(device_.parameters(), parameters_.seed_, device_.unit_cell_size());

  
  if(parameters_.kernel_choice_ == 0)
    kernel_   = new None();
  else if(parameters_.kernel_choice_==1)
    kernel_   = new Jackson();
  
}



KPM_base::~KPM_base(){

/*------------Delete everything--------------*/

  delete []rand_vec_;
  delete []dmp_op_;
   
  delete kernel_;
  delete cap_;
  delete vec_base_;
}

type cdot(type* vec_1, type* vec_2, int D){
  type result = 0;
  
  //#pragma omp parallel for
  for(int i = 0; i<D; i++)
    result += conj(vec_1[i]) * vec_2[i];
  
  return result;
};


void KPM_base::compute(){

  time_station_2 solver_station;
  solver_station.start();
  

  //This should be done outside the solver loop
  //----------------Initializing the Device---------------//
  //------------------------------------------------------//
  time_station_2 hamiltonian_setup_time;
  hamiltonian_setup_time.start();

  initialize_device();  

  hamiltonian_setup_time.stop("    Time to setup the Hamiltonian:            ");
  std::cout<<std::endl;

  //------------------------------------------------------//
  //------------------------------------------------------//  



  //And if the previous step is outside, then the following can be done as
  //an initialization, simply
  //-------------------Allocating memory------------------//
  //------------------------------------------------------//  
  time_station_2 allocation_time;
  allocation_time.start();

  allocate_memory();
  
  allocation_time.stop("\n \nAllocation time:            ");
  //------------------------------------------------------//
  //------------------------------------------------------//  



  //This should just be initialization of the CAP object
  //-------------------This shouldnt be heeere--------------//
  int W      = device().parameters().W_,
      C      = device().parameters().C_,
      LE     = device().parameters().LE_,
    DIM    = device().parameters().DIM_,
    SUBDIM    = device().parameters().SUBDIM_;

  r_type a       = parameters_.a_,
         E_min   = parameters_.E_min_,
         eta     = parameters_.eta_;
    
  E_min /= a;
  eta   /= a;


  dmp_op_ = new r_type[ DIM ];
  rand_vec_ = new type[ SUBDIM ];

  cap().create_CAP(W, C, LE,  dmp_op_);
  device().damp(dmp_op_);
  //-------------------This shouldnt be heeere--------------//  


  
  int R = parameters().R_,
      D = parameters().dis_real_;



  
  for(int d = 1; d <= D; d++){

    device_.update_dis( dmp_op_);
    
    for(int r = 1; r <= R; r++){


      time_station_2 randVec_time;
      randVec_time.start();
      
      std::cout<<std::endl<< std::to_string( ( d - 1 ) * R + r)+"/"+std::to_string( D * R )+"-Vector/disorder realization;"<<std::endl;

      vec_base_->generate_vec_im( rand_vec_, r);       
      device_.rearrange_initial_vec( rand_vec_ ); //very hacky

      if(dynamic_cast<Graphene_KaneMele*>(&device_) && !device_.isKspace() && parameters_.base_choice_ == 4)
	device_.to_kSpace(rand_vec_, rand_vec_, 1);

      
      if(dynamic_cast<Graphene_KaneMele*>(&device_) && device_.isKspace() && parameters_.base_choice_ == 1 )
      	device_.Uk_ket(rand_vec_, rand_vec_);	
      
      
      


      compute_rand_vec( ( d - 1 ) * R + r );

            
      randVec_time.stop("       Total RandVec time:         ");
      std::cout<<std::endl;

  }


  
  solver_station.stop("Total case execution time:              ");
  }
};






void KPM_DOS_solver::allocate_memory(){
  //cheb_vectors_ = new Chebyshev_states<State<type>>( device() );
}




void KPM_DOS_solver::compute_rand_vec(int r){
  int M   = parameters().M_;

  Chebyshev_states cheb_vectors_(device());
  
  moments_r_ = std::vector<type>(parameters().M_, 0.0);

  std::vector<type> l_r_vec(rand_vec(), rand_vec()+device().parameters().DIM_);

  device().projector(l_r_vec.data());
  
  
  cheb_vectors_.reset( rand_vec() );
      
//=================================KPM Step 0======================================//
  moments_r_[0] = cdot ( l_r_vec.data() , (cheb_vectors_)(0), device().parameters().DIM_ );


  
//=================================KPM Step 1======================================//       
  cheb_vectors_.update();
  moments_r_[1] = cdot ( l_r_vec.data() , (cheb_vectors_)(1), device().parameters().DIM_ );


  
//=================================KPM Steps 2 and on==============================//
  for( int m = 2; m < M; m++ ){
    cheb_vectors_.update();
    moments_r_[m] = cdot ( l_r_vec.data(), (cheb_vectors_)(2), device().parameters().DIM_ );
  }



  
  for(int m=0;m<M;m++)
    moments_acc_[m] = ( moments_r_[m] + double(r-1) * moments_acc_[m] ) / r;
    
  
 

  
  output_.update_data(moments_r_, moments_acc_, r);
}



DOS_output::DOS_output(device_vars& device_parameters, solver_vars& parent_solver_vars):device_parameters_(device_parameters), parameters_(parent_solver_vars){
  int nump = parameters_.num_p_,
    D = parameters_.dis_real_,
    R = parameters_.R_;
  
  E_points_.resize(nump);

  conv_R_max_.resize(D*R);
  conv_R_av_.resize(D*R);

  partial_result_.resize(nump);
  r_data_.resize(nump);

  partial_result_=std::vector<r_type>(nump, 0.0);
  r_data_=std::vector<r_type>(nump, 0.0);
  
  if(parameters_.kernel_choice_ == 0)
    kernel_   = new None();
  else if(parameters_.kernel_choice_==1)
    kernel_   = new Jackson();

  
  
  for(int k=0; k<nump;k++)
    E_points_[k] = cos(M_PI * ( k + 0.5 ) / nump );



  
  std::string filename = parent_solver_vars.filename_;


  mkdir( ( "./" + filename ).c_str(), 0755);
  mkdir( ( "./" + filename + "/"  + "vecs" ).c_str(), 0755);
  

  const std::string sourceFile = "./SimData.dat";  // Source file path
  const std::string destinationFile =  "./" + filename + "/SimData.dat" ;  // Destination file path

  fs::copy(sourceFile, destinationFile, fs::copy_options::overwrite_existing);

};


void DOS_output::update_data(std::vector<type>& moments_r, std::vector<type>& moments_acc, int r){

  const std::complex<double> im(0,1);  

  std::string run_dir = parameters_.run_dir_,
              filename = parameters_.filename_;

 

  
  int M = parameters_.M_,
      nump = parameters_.num_p_;
  
  int SUBDIM = device_parameters_.SUBDIM_;    

  r_type a = parameters_.a_,
         b = parameters_.b_;

  r_type omega =  SUBDIM / ( a * M_PI );//Dimensional and normalizing constant


  
  std::vector<r_type> new_partial_result(nump, 0.0),
    new_r_data(nump, 0.0);

  fftw_plan plan;      
  r_type* input,
        * output;

  
  output = ( r_type* ) fftw_malloc( sizeof( r_type ) * nump );    
  input  = ( r_type* ) fftw_malloc( sizeof( r_type ) * nump );    

  for(int i = 0; i < nump;i++){
    input[i]=0.0;
    output[i]=0.0;
  }
  
  plan = fftw_plan_r2r_1d(nump, input, output, FFTW_REDFT01, FFTW_ESTIMATE);


  
  
  for(int m = 0; m < M; m++)
    input[m] = ( 2 - ( m == 0 ) ) * kernel_->term(m,M) * moments_r[m].real();
  
  //fftw_execute( plan ); 

  //  for(int i = 0; i < nump;i++)
  //  r_data_[i] = output[i] / sqrt( 1.0 - E_points_[i] * E_points_[i] );

  for(int i = 0; i < nump;i++){
    for(int m = 0; m < M; m++)
      new_r_data[i] += input[m] * (std::cos( m*(i + 0.5) * M_PI /nump));//output[i] / sqrt( 1.0 - E_points_[i] * E_points_[i] );//

    new_r_data[i] /=  sqrt( 1.0 - E_points_[i] * E_points_[i] );
  }
  
  
  

  
  for(int m = 0; m < M; m++)
    input[m] = ( 2 - ( m == 0 ) ) * kernel_->term(m,M) * moments_acc[m].real();
  
  //fftw_execute( plan ); 

  for(int i = 0; i < nump;i++){
    for(int m = 0; m < M; m++)
      new_partial_result[i] += input[m] * (std::cos( m*(i + 0.5) * M_PI /nump));//output[i] / sqrt( 1.0 - E_points_[i] * E_points_[i] );//

    new_partial_result[i] /=  sqrt( 1.0 - E_points_[i] * E_points_[i] );
  }

  
  fftw_free(output);
  fftw_free(input);
  fftw_destroy_plan(plan);
  



  
  //R convergence analysis    
  r_type tmp = 1, max = 0, av=0;

  for(int e = 0; e < nump; e++){

    tmp = partial_result_ [ e ] ;
    if( r > 1 ){
      tmp = std::abs( ( new_partial_result [ e ] - partial_result_ [e] ) / partial_result_ [e] ) ;
      if(tmp > max)
        max = tmp;

      av += tmp / nump ;
    }
  }


  
  
  if( r > 1 ){
    conv_R_max_[ ( r - 1 ) ] = max;
    conv_R_av_ [ ( r - 1 ) ] = av;
  }

  partial_result_ = new_partial_result;
  r_data_ = new_r_data;  



 
    
  std::ofstream dataR;
  dataR.open("./" + filename+"/"+run_dir+"vecs/r"+std::to_string(r)+".dat" );

  for(int e=0;e<nump;e++)  
    dataR<< a * E_points_[e] - b<<"  "<< omega * r_data_[e]<<"  "<<  omega * new_partial_result [e] <<std::endl;

  dataR.close();
  


  
  std::ofstream dataP;
  dataP.open("./" + filename+"/currentResult.dat");

  for(int e=0;e<nump;e++)  
    dataP<< a * E_points_[e] - b<<"  "<<  omega *  partial_result_ [e] <<std::endl;

  dataP.close();



  
  
  std::ofstream data;
  data.open("./" + filename+"/conv_R.dat");

  for(int l = 1; l < r; l++)  
    data<< l <<"  "<< conv_R_max_[ ( l - 1 ) ]<<"  "<< conv_R_av_[ ( l - 1 ) ] <<std::endl;

  data.close();
  

 
  plot_data("./" + filename,"");


}


void DOS_output::plot_data(const std::string& run_dir, const std::string& filename){
        //VIEW commands
  
     std::string exestring=
         "gnuplot<<EOF                                               \n"
         "set encoding utf8                                          \n"
         "set terminal pngcairo enhanced                             \n"

         "unset key  \n"

       
         "set output '"+run_dir+"/currentResult.png'                \n"

       
         "set xlabel 'E[eV]'                                               \n"
         "set ylabel  'DOS [a.U.]'                                           \n"
         

        "plot '"+run_dir+"/currentResult.dat'  using 1:2 w p ls 7 ps 0.25 lc 2;  \n"
        "EOF";
     
      char exeChar[exestring.size() + 1];
      strcpy(exeChar, exestring.c_str());    
      if(system(exeChar)){};


}
