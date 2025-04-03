#include "Kubo_solver_FFT.hpp"
#include "../time_station.hpp"

#include <sys/stat.h>
#include <sys/types.h>

#include <filesystem>

namespace fs = std::filesystem;
  


Kubo_solver_FFT_postProcess::Kubo_solver_FFT_postProcess(Kubo_solver_FFT& parent_solver): parent_solver_(parent_solver){
  int nump = parent_solver_.parameters().num_p_,
    D = parent_solver_.parameters().dis_real_,
    R = parent_solver_.parameters().R_;

  
  E_points_.resize(nump);

  conv_R_max_.resize(D*R);
  conv_R_av_.resize(D*R);

  prev_partial_result_.resize(nump);
  
  for(int k=0; k<nump;k++){
    E_points_[k] = cos(M_PI * ( 2 * k + 0.5 ) / nump );
  }



  
  std::string filename = parent_solver_.parameters().filename_;


  mkdir( ( "./" + filename ).c_str(), 0755);
  mkdir( ( "./" + filename + "/"  + "vecs" ).c_str(), 0755);
  

  const std::string sourceFile = "./SimData.dat";  // Source file path
  const std::string destinationFile =  "./" + filename + "/SimData.dat" ;  // Destination file path

  fs::copy(sourceFile, destinationFile, fs::copy_options::overwrite_existing);

};



void Kubo_solver_FFT_postProcess::operator()(const std::vector<type>& final_data, const std::vector<type>& r_data, int r){
  if(parent_solver_.simulation_formula() == KUBO_GREENWOOD)
    Greenwood_postProcess(final_data, r_data, r);

  if(parent_solver_.simulation_formula() == KUBO_BASTIN )
    Bastin_postProcess(final_data, r_data, r);

  if( parent_solver_.simulation_formula() == KUBO_SEA )
    Sea_postProcess(final_data, r_data, r);

};


void Kubo_solver_FFT_postProcess::eta_CAP_correct(std::vector<r_type>& E_points, std::vector<type>& r_data){
  int nump = parent_solver_.parameters().num_p_;
  
  for(int e=0; e < nump; e++ )
    r_data[e] *= sin( acos( E_points[e] ) );
}


void Kubo_solver_FFT_postProcess::integration_linqt(const std::vector<r_type>& E_points, const std::vector<r_type>& integrand, std::vector<r_type>& result){

  int nump = parent_solver_.parameters().num_p_;

  double acc = 0;
  for( int i=0; i < nump-1; i++)
  {
     const double denerg = E_points[i+1]-E_points[i];
     acc +=integrand[i]*denerg;
     result[i] = acc;
  }
}

  


void Kubo_solver_FFT_postProcess::integration(const std::vector<r_type>& E_points, const std::vector<r_type>& integrand, std::vector<r_type>& result){

  int M     = parent_solver_.parameters().M_,
    nump = parent_solver_.parameters().num_p_;
  
  r_value_t edge = 1.0 - parent_solver_.parameters().edge_;

  //#pragma omp parallel for 
  for(int k=0; k<nump - int( M * edge / 4.0 ); k++ ){  //At the very edges of the energy plot the weight function diverges, hence integration should start a little after and a little before the edge.
                                       //The safety factor guarantees that the conductivity is zero way before reaching the edge. In fact, the safety factor should be used to determine
                                      //the number of points to be ignored in the future;
    for(int j=k; j<nump-int(M*edge/4.0); j++ ){//IMPLICIT PARTITION FUNCTION. Energies are decrescent with e (bottom of the band structure is at e=M);
      r_value_t ej  = E_points[j],
	ej1      = E_points[j+1],
	de       = ej-ej1,
        integ    = ( integrand[j+1] + integrand[j] )/2;     
      
      result[k] +=  de * integ;
    }
  }
}


  
void Kubo_solver_FFT_postProcess::partial_integration(const std::vector<r_type>& E_points, const std::vector<r_type>& integrand, std::vector<r_type>& data){

  int M     = parent_solver_.parameters().M_,
      end   = M * 0.55,
      start = M * 0.45;
  
  //#pragma omp parallel for 
  for( int k = start; k < end; k++ ){  //At the very edges of the energy plot the weight function diverges, hence integration should start a little after and a little before the edge.
                                       //The safety factor guarantees that the conductivity is zero way before reaching the edge. In fact, the safety factor should be used to determine
                                       //the number of points to be ignored in the future;
    for(int j=k; j<end; j++ ){//IMPLICIT PARTITION FUNCTION. Energies are decrescent with e (bottom of the band structure is at e=M);
        r_value_t ej  = E_points[j],
	ej1      = E_points[j+1],
	de       = ej-ej1,
        integ    = ( integrand[j+1] + integrand[j] ) / 2.0;     
      
      data[k] +=  de * integ;
    }
  }
}
     

void Kubo_solver_FFT_postProcess::rearrange_crescent_order( std::vector<r_type>& rearranged){//The point choice of the FFT has unconvenient ordering for file saving and integrating; This fixes that.
  int nump = parent_solver_.parameters().num_p_;
  std::vector<r_type> original(nump);

  for( int k = 0; k < nump; k++ )
    original[k] = rearranged[k];

  
  for( int k = 0; k < nump / 2; k++ ){
    rearranged[ 2 * k ]   = original[ k ];
    rearranged[ 2 * k + 1 ] = original[ nump - k - 1 ]; 
  }
  
  for( int k=0; k < nump / 2; k++ ){
    r_type tmp = rearranged[ k ]; 
    rearranged[ k ]   = rearranged[ nump-k-1 ];
    rearranged[ nump-k-1 ] = tmp;    
  }  
}


void Kubo_solver_FFT_postProcess::rearrange_crescent_order_2(std::vector<r_type>& E_points, std::vector<r_type>& data) {
    // Create a vector of indices from 0 to nump - 1
    int nump = E_points.size();
    std::vector<int> indices(nump);
    
    for (int i = 0; i < nump; ++i) {
        indices[i] = i;
    }

    // Sort the indices based on the values of E_points
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return E_points[a] < E_points[b];
    });

    // Create temporary vectors to store the sorted versions
    std::vector<r_type> sorted_E_points(nump);
    std::vector<r_type> sorted_data(nump);

    // Reorder both vectors based on sorted indices
    for (int i = 0; i < nump; ++i) {
        sorted_E_points[i] = E_points[indices[i]];
        sorted_data[i] = data[indices[i]];
    }

    // Move sorted values back to original vectors
    E_points = std::move(sorted_E_points);
    data = std::move(sorted_data);
}

void Kubo_solver_FFT_postProcess::Sea_postProcess(const std::vector<type>& final_data, const std::vector<type>& r_data, int r){

  const std::complex<double> im(0,1);  

  
  std::string run_dir = parent_solver_.parameters().run_dir_,
              filename = parent_solver_.parameters().filename_;
  
  
  int nump = parent_solver_.parameters().num_p_;
  size_t SUBDIM = parent_solver_.device().parameters().SUBDIM_;    

  r_type a = parent_solver_.parameters().a_,
    b = parent_solver_.parameters().b_,
    sysSubLength = parent_solver_.device().sysSubLength();

  
  //  r_type omega = DIM/( a * a );//* sysSubLength * sysSubLength );//Dimensional and normalizing constant
  r_type omega = 2.0 * SUBDIM/( a * a * sysSubLength * sysSubLength ) /* ( 2 * M_PI )*/;//Dimensional and normalizing constant. The minus is due to the vel. op being conjugated.
  //r_value_t tmp, max=0, av=0;

  
  std::vector<r_type>
    rearranged_E_points(nump),
    integrand(nump),
    rvec_integrand(nump),
    partial_result(nump),
    rvec_partial_result(nump);

  for(int e=0; e<nump; e++){
    rearranged_E_points[e] = E_points_[e];
    integrand[e]           = 0.0;
    rvec_integrand[e]      = 0.0;
    partial_result[e]      = 0.0;
    rvec_partial_result[e] = 0.0;
  }   


  //rearrange_crescent_order(rearranged_E_points);

  /*
  When introducing a const. eta with modified polynomials, the result is equals to that of a
  simulation with regular polynomials and an variable eta_{var}=eta*sin(acos(E)). The following
  heuristical correction greatly improves the result far from the CNP to match that of the
  desired regular polys and const. eta.
  */
  



  

    

  //Keeping just the real part of E*p(E)+im*sqrt(1-E^2)*w(E) yields the Kubo-Bastin integrand:
  for(int k = 0; k < nump; k++){
    integrand[k]  = -E_points_[k] * imag( final_data[ k ] ) - ( sqrt(1.0 - E_points_[ k ] * E_points_[ k ] ) * imag( final_data[ k + nump ] ) );
    integrand[k] *= 1.0 / pow( (1.0 - E_points_[k]  * E_points_[k] ), 2.0);
    integrand[k] *=  omega ; 

    rvec_integrand[k]  = -E_points_[k] * imag( r_data[ k ] ) - ( sqrt(1.0 - E_points_[ k ] * E_points_[ k ] ) * imag( r_data[ k + nump ] ) );
    rvec_integrand[k] *= 1.0 / pow( (1.0 - E_points_[k]  * E_points_[k] ), 2.0);
    rvec_integrand[k] *=  omega ; 
  }

  
  rearrange_crescent_order_2(rearranged_E_points, integrand);
  //rearrange_crescent_order(integrand);
  rearrange_crescent_order(rvec_integrand);

  


  time_station time_integration;
      
  integration(rearranged_E_points, rvec_integrand, rvec_partial_result);  
  integration(rearranged_E_points, integrand, partial_result);    

  time_integration.stop("       Integration time:           ");

  

  /*
  if( parameters_.eta_!=0 ){
    eta_CAP_correct(rearranged_E_points, partial_result);
    eta_CAP_correct(rearranged_E_points, rvec_partial_result);
  }
  */

    r_type tmp = 1, max = 0, av=0;
  //R convergence analysis  
  for(int e = 0; e < nump; e++){

    tmp = partial_result [ e ] ;
    if( r > 1 ){
      tmp = std::abs( ( partial_result [ e ] - prev_partial_result_ [e] ) / prev_partial_result_ [e] ) ;
      if(tmp > max)
        max = tmp;

      av += tmp / nump ;
    }
  }

  
  if( r > 1 ){
    conv_R_max_[ ( r - 1 ) ] = max;
    conv_R_av_ [ ( r - 1 ) ] = av;
  }


  


     
  prev_partial_result_=partial_result;
  
  std::ofstream dataR;
  dataR.open("./" + filename+"/"+run_dir+"vecs/r"+std::to_string(r)+".dat");

  for(int e=0;e<nump;e++)  
    dataR<< a * rearranged_E_points[e] - b<<"  "<< rvec_partial_result [e]<<"  "<<  partial_result [e] <<std::endl;

  dataR.close();
  


  
  std::ofstream dataP;
  dataP.open("./" + filename+"/currentResult.dat");

  for(int e=0;e<nump;e++)  
    dataP<< a * rearranged_E_points[e] - b<<"  "<<  partial_result [e] <<std::endl;

  dataP.close();



  
  
  std::ofstream data;
  data.open("./" + filename+"/conv_R.dat");

  for(int l = 1; l < r; l++)  
    data<< l <<"  "<< conv_R_max_[ ( l - 1 ) ]<<"  "<< conv_R_av_[ ( l - 1 ) ] <<std::endl;

  data.close();
  

  
  
  
  std::ofstream data2;
  data2.open( "./" + filename + "Bastin_integrand.dat");

  for(int e=0;e<nump;e++)  
    data2<< a * rearranged_E_points[e] - b<<"  "<<  integrand[e] <<std::endl;
  
  data2.close();


  plot_data("./" +filename, "");

}



void Kubo_solver_FFT_postProcess::Bastin_postProcess(const std::vector<type>& final_data, const std::vector<type>& r_data, int r){

  const std::complex<double> im(0,1);  

  
  std::string run_dir = parent_solver_.parameters().run_dir_,
              filename = parent_solver_.parameters().filename_;
  
  
  int nump = parent_solver_.parameters().num_p_;
  size_t SUBDIM = parent_solver_.device().parameters().SUBDIM_;    

  r_type a = parent_solver_.parameters().a_,
    b = parent_solver_.parameters().b_,
    sysSubLength = parent_solver_.device().sysSubLength();

  
  //  r_type omega = DIM/( a * a );//* sysSubLength * sysSubLength );//Dimensional and normalizing constant
  r_type omega = 2.0 * SUBDIM/( a * a * sysSubLength * sysSubLength ) /* ( 2 * M_PI )*/;//Dimensional and normalizing constant. The minus is due to the vel. op being conjugated.(REMOVED)
  //r_value_t tmp, max=0, av=0;

  
  std::vector<r_type>
    rearranged_E_points(nump),
    integrand(nump),
    rvec_integrand(nump),
    partial_result(nump),
    rvec_partial_result(nump);

  for(int e=0; e<nump; e++){
    rearranged_E_points[e] = E_points_[e];
    integrand[e]           = 0.0;
    rvec_integrand[e]      = 0.0;
    partial_result[e]      = 0.0;
    rvec_partial_result[e] = 0.0;
  }   


  //rearrange_crescent_order(rearranged_E_points);

  /*
  When introducing a const. eta with modified polynomials, the result is equals to that of a
  simulation with regular polynomials and an variable eta_{var}=eta*sin(acos(E)). The following
  heuristical correction greatly improves the result far from the CNP to match that of the
  desired regular polys and const. eta.
  */
  



  

    

  //Keeping just the real part of E*p(E)+im*sqrt(1-E^2)*w(E) yields the Kubo-Bastin integrand:
  for(int k = 0; k < nump; k++){
    integrand[k]  = E_points_[k] * real( final_data[ k ] ) - ( sqrt(1.0 - E_points_[ k ] * E_points_[ k ] ) * imag( final_data[ k + nump ] ) );
    integrand[k] *= 1.0 / pow( (1.0 - E_points_[k]  * E_points_[k] ), 2.0);
    integrand[k] *=  omega ; 

    rvec_integrand[k]  = E_points_[k] * real( r_data[ k ] ) - ( sqrt(1.0 - E_points_[ k ] * E_points_[ k ] ) * imag( r_data[ k + nump ] ) );
    rvec_integrand[k] *= 1.0 / pow( (1.0 - E_points_[k]  * E_points_[k] ), 2.0);
    rvec_integrand[k] *=  omega ; 
  }

  
  rearrange_crescent_order_2(rearranged_E_points, integrand);
  //rearrange_crescent_order(integrand);
  rearrange_crescent_order(rvec_integrand);

  


  time_station time_integration;
      
  integration(rearranged_E_points, rvec_integrand, rvec_partial_result);  
  integration(rearranged_E_points, integrand, partial_result);    

  time_integration.stop("       Integration time:           ");

  

  /*
  if( parameters_.eta_!=0 ){
    eta_CAP_correct(rearranged_E_points, partial_result);
    eta_CAP_correct(rearranged_E_points, rvec_partial_result);
  }
  */

    r_type tmp = 1, max = 0, av=0;
  //R convergence analysis  
  for(int e = 0; e < nump; e++){

    tmp = partial_result [ e ] ;
    if( r > 1 ){
      tmp = std::abs( ( partial_result [ e ] - prev_partial_result_ [e] ) / prev_partial_result_ [e] ) ;
      if(tmp > max)
        max = tmp;

      av += tmp / nump ;
    }
  }

  
  if( r > 1 ){
    conv_R_max_[ ( r - 1 ) ] = max;
    conv_R_av_ [ ( r - 1 ) ] = av;
  }


  
  prev_partial_result_=partial_result;


  std::ofstream dataR;
  dataR.open("./" + filename+"/"+run_dir+"vecs/r"+std::to_string(r)+".dat" );

  for(int e=0;e<nump;e++)  
    dataR<< a * rearranged_E_points[e] - b<<"  "<< rvec_partial_result [e]<<"  "<<  partial_result [e] <<std::endl;

  dataR.close();
  


  
  std::ofstream dataP;
  dataP.open("./" + filename+"/currentResult.dat");

  for(int e=0;e<nump;e++)  
    dataP<< a * rearranged_E_points[e] - b<<"  "<<  partial_result [e] <<std::endl;

  dataP.close();



  
  
  std::ofstream data;
  data.open("./" + filename+"/conv_R.dat");

  for(int l = 1; l < r; l++)  
    data<< l <<"  "<< conv_R_max_[ ( l - 1 ) ]<<"  "<< conv_R_av_[ ( l - 1 ) ] <<std::endl;

  data.close();
  

  
  
  
  std::ofstream data2;
  data2.open( "./" + filename + "Bastin_integrand.dat");

  for(int e=0;e<nump;e++)  
    data2<< a * rearranged_E_points[e] - b<<"  "<<  integrand[e] <<std::endl;
  
  data2.close();


  plot_data("./" + filename,"");

}




void Kubo_solver_FFT_postProcess::Greenwood_postProcess(const std::vector<type>& final_data, const std::vector<type>& r_data, int r){

  const std::complex<double> im(0,1);  

  std::string run_dir = parent_solver_.parameters().run_dir_,
              filename = parent_solver_.parameters().filename_;

 

  
  int nump = parent_solver_.parameters().num_p_;
  //    R = parameters_.R_,
  //  D = parameters_.dis_real_;
  
  size_t SUBDIM = parent_solver_.device().parameters().SUBDIM_;    

  r_type a = parent_solver_.parameters().a_,
         b = parent_solver_.parameters().b_,
         sysSubLength = parent_solver_.device().sysSubLength();
  
  r_type omega = 2.0 * SUBDIM/( a * a * sysSubLength * sysSubLength ) /* ( 2 * M_PI )*/;//Dimensional and normalizing constant (MINUS REMOVED)
  
  //  r_value_t tmp, max=0, av=0;

  std::vector<r_type>
    rearranged_E_points(nump),
    partial_result(nump),
    rvec_partial_result(nump);

  
  for(int e=0; e<nump; e++){
    rearranged_E_points[e] =  E_points_[e] ;
    partial_result[e] = 0.0;
    rvec_partial_result[e] = 0.0;
  }

  /*When introducing a const. eta with modified polynomials, the result is equals to that of a
  simulation with regular polynomials and an variable eta_{var}=eta*sin(acos(E)). The following
  heuristical correction greatly improves the result far from the CNP to match that of the
  desired regular polys and const. eta.*/
      
  
  for(int k=0; k < nump; k++){
    rvec_partial_result[k] = omega * real( r_data[k] )     / (1.0 - E_points_[k] * E_points_[k] ); //The minus sign represents the conjugation of vx being applied to the bra
    partial_result[k]      = omega * real( final_data[k] ) / (1.0 - E_points_[k] * E_points_[k] );
  }
  
  rearrange_crescent_order(rearranged_E_points);
  rearrange_crescent_order(partial_result);
  rearrange_crescent_order(rvec_partial_result);


  /*
  if( parameters_.eta_ != 0 ){
    eta_CAP_correct(rearranged_E_points, partial_result);
    eta_CAP_correct(rearranged_E_points, rvec_partial_result);
  }
  */


  r_type tmp = 1, max = 0, av=0;
  //R convergence analysis  
  for(int e = 0; e < nump; e++){

    tmp = partial_result [ e ] ;
    if( r > 1 ){
      tmp = std::abs( ( partial_result [ e ] - prev_partial_result_ [e] ) / prev_partial_result_ [e] ) ;
      if(tmp > max)
        max = tmp;

      av += tmp / nump ;
    }
  }

  
  if( r > 1 ){
    conv_R_max_[ ( r - 1 ) ] = max;
    conv_R_av_ [ ( r - 1 ) ] = av;
  }

  prev_partial_result_ = partial_result;
  
  
  std::ofstream dataR;
  dataR.open("./" + filename+"/"+run_dir+"vecs/r"+std::to_string(r)+".dat" );

  for(int e=0;e<nump;e++)  
    dataR<< a * rearranged_E_points[e] - b<<"  "<< rvec_partial_result [e]<<"  "<<  partial_result [e] <<std::endl;

  dataR.close();
  


  
  std::ofstream dataP;
  dataP.open("./" + filename+"/currentResult.dat");

  for(int e=0;e<nump;e++)  
    dataP<< a * rearranged_E_points[e] - b<<"  "<<  partial_result [e] <<std::endl;

  dataP.close();



  
  
  std::ofstream data;
  data.open("./" + filename+"/conv_R.dat");

  for(int l = 1; l < r; l++)  
    data<< l <<"  "<< conv_R_max_[ ( l - 1 ) ]<<"  "<< conv_R_av_[ ( l - 1 ) ] <<std::endl;

  data.close();
  

 
  plot_data("./" + filename,"");


  
}








void Kubo_solver_FFT_postProcess::plot_data(const std::string& run_dir, const std::string& filename){
        //VIEW commands
  
     std::string exestring=
         "gnuplot<<EOF                                               \n"
         "set encoding utf8                                          \n"
         "set terminal pngcairo enhanced                             \n"

         "unset key  \n"

         "set output '"+run_dir+"/currentResult.png'                \n"

         "set xlabel 'E[eV]'                                               \n"
         "set ylabel  'G [2e^2/h]'                                           \n"
         
        "plot '"+run_dir+"/currentResult.dat'  using 1:2 w p ls 7 ps 0.25 lc 2;  \n"
         "EOF";
     
      char exeChar[exestring.size() + 1];
      strcpy(exeChar, exestring.c_str());    
      if(system(exeChar)){};


}


