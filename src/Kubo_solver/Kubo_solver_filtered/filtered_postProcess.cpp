
#include "../../complex_op.hpp"
#include "Kubo_solver_filtered.hpp"

#include "../time_station.hpp"




void Kubo_solver_filtered::rearrange_crescent_order( r_type* rearranged){//The point choice of the FFT has unconvenient ordering for file saving and integrating; This fixes that.
  int nump = parameters_.num_p_;
  std::vector<r_type> original(nump);

  for( int k = 0; k < nump; k++ )
    original[ k ] = rearranged[ k ];

  for(int e = 0; e<nump/2; e++){

    rearranged[e] = original[ nump / 2 + e ];
    rearranged[ nump / 2 + e ] = original[e];
  }
  
}



void Kubo_solver_filtered::integration_linqt(const r_type* E_points, const r_type* integrand, r_type* result){

  int nump = parameters_.num_p_;

  double acc = 0;
  for( int i=0; i < nump; i++)
  {
     const double denerg = E_points[i+1]-E_points[i];
     acc +=integrand[i]*denerg;
     result[i] = acc;
  }
}



void Kubo_solver_filtered::interpolated_integration(const r_type* E_points, const r_type* integrand,  r_type* result){

  int nump = parameters_.num_p_;

  r_type interpolated_E_points[2*nump],
    interpolated_integrand[2*nump],
    full_result[2*nump];


  
  for( int i=0; i < nump; i++)
  {
      interpolated_integrand[2*i] = integrand[i];
      interpolated_E_points[2*i] = E_points[i];
    
      interpolated_E_points[2*i+1] = (E_points[i+1] + E_points[i])/2.0;
      interpolated_integrand[2*i+1] = (integrand[i+1]+integrand[i])/2.0;
  }

  double acc = 0;
  for( int i=0; i < 2*nump; i++)
  {
     const double denerg = interpolated_E_points[i+1]-interpolated_E_points[i];
     acc +=interpolated_integrand[i]*denerg;
     full_result[i] = acc;
  }

  
  for( int i=0; i < nump; i++)
      result[i] = full_result[2*i];
  
  
}


/*
void Kubo_solver_filtered::integration_linqt(const r_type* E_points, const r_type* integrand, r_type* result){

    int nump = parameters_.num_p_;

    // Check if nump is odd, as Simpson's rule requires an even number of intervals
    if (nump % 2 == 0) {
        nump -= 1; // Ensure we have an odd number of points
    }

    double acc = 0;
    result[0] = 0; // Initial result for the first point

    for (int i = 0; i < nump - 2; i ++) {
        const double h = E_points[i+2] - E_points[i];  // Step size over 3 points
        acc += (h / 6) * ( integrand[i] + 4 * integrand[i+1] + integrand[i+2] );  // Simpson's rule
        result[i+2] = acc/2;
    }
    
    // Handle the last interval if the number of points is even
    if (nump % 2 == 0) {
        const double h = E_points[nump] - E_points[nump-1];
        acc += 0.5 * h * (integrand[nump-1] + integrand[nump]);
        result[nump] = acc;
    }
}
*/

/*
void Kubo_solver_filtered::integration_linqt(const r_type* E_points, const r_type* integrand, r_type* result){

    int nump = parameters_.num_p_;

    double acc = 0;

    // Iterate through the points using Simpson's rule
    for (int i = 0; i < nump - 2; i +=2) {

        // Calculate step sizes between the points
        double h1 = E_points[i+1] - E_points[i];   // Step between E_points[i] and E_points[i+1]
        double h2 = E_points[i+2] - E_points[i+1]; // Step between E_points[i+1] and E_points[i+2]

        // Simpson's rule for non-equidistant points
        double h = (h1 + h2) / 2.0;  // Average step size
        double simpson_integral = (integrand[i] * (h2 * (2 * h1 + h2)) +
                                   4 * integrand[i+1] * (h1 + h2) +
                                   integrand[i+2] * (h1 * (h1 + 2 * h2))) / (6.0 * (h1 + h2));

        // Accumulate the result
        acc += h * simpson_integral;
        result[i] = acc;   // Store the result at current index
    }

    // If the number of points is odd, apply trapezoidal rule for the last interval
    if (nump % 2 == 1) {
        int last_point = nump - 2;
        double denerg = E_points[last_point + 1] - E_points[last_point];  // Step between the last two points
        double trap_integral = (integrand[last_point] + integrand[last_point + 1]) / 2.0 * denerg;

        // Accumulate and store the final result
        acc += trap_integral;
        result[last_point] = acc;
    }
}
*/




void Kubo_solver_filtered::update_data_Bastin(r_type E_points[], type r_data[], type final_data[], r_type conv_R[], int r, std::string run_dir, std::string filename){

  const std::complex<double> im(0,1);  
  
  
  int nump = parameters_.num_p_,
    R = parameters_.R_,
    D = parameters_.dis_real_;
  
  int SUBDIM = device_.parameters().SUBDIM_;    

  r_type a = parameters_.a_,
    b = parameters_.b_,
    sysSubLength = device_.sysSubLength();
  
  int decRate = filter_.parameters().decRate_;   
  
  r_type omega = 2.0 * decRate * decRate * SUBDIM/( a * a * sysSubLength * sysSubLength ) /* ( 2 * M_PI )*/;//Dimensional and normalizing constant. The minus is due to the vel. op being conjugated.
  //DIM/( a * a );//* sysSubLength * sysSubLength );//Dimensional and normalizing constant
  
  //r_value_t tmp, max=0, av=0;

  r_type
    rearranged_E_points[nump],
    integrand[nump],
    rvec_integrand[nump],
    partial_result[nump],
    rvec_partial_result[nump];

  for(int e=0; e<nump; e++){
    rearranged_E_points[e] = E_points[e];
    integrand[e]           = 0.0;
    rvec_integrand[e]      = 0.0;
    partial_result[e]      = 0.0;
    rvec_partial_result[e] = 0.0;
  }   

  
    


  
  for( int e = 0; e < 2*nump; e++ ){  
    //prev_partial_result[e] = final_data[e];
    final_data[e] = ( final_data [e] * (r-1.0) +  r_data[e] ) / r;
  }


  

    

  //Keeping just the real part of E*p(E)+im*sqrt(1-E^2)*w(E) yields the Kubo-Bastin integrand:
  for(int k = 0; k < nump; k++){
    
    integrand[k]  = E_points[k] * real( final_data[ k ] ) - ( sqrt(1.0 - E_points[ k ] * E_points[ k ] ) * imag( final_data[ k + nump ] ) );
    integrand[k] *= 1.0 / pow( (1.0 - E_points[k]  * E_points[k] ), 2.0);
    integrand[k] *=  omega ; 

    rvec_integrand[k]  = E_points[k] * real( r_data[ k ] ) - ( sqrt(1.0 - E_points[ k ] * E_points[ k ] ) * real( r_data[ k + nump ] ) );
    rvec_integrand[k] *= 1.0 / pow( (1.0 - E_points[k]  * E_points[k] ), 2.0);
    rvec_integrand[k] *=  omega ; 
  }

  rearrange_crescent_order(rearranged_E_points);
  rearrange_crescent_order(integrand);
  rearrange_crescent_order(rvec_integrand);

  


  time_station time_integration;
      
  integration_linqt(rearranged_E_points, rvec_integrand, rvec_partial_result);  
  /*integration_linqt*/ interpolated_integration(rearranged_E_points, integrand, partial_result);    

  time_integration.stop("       Integration time:           ");

  

  /*
  if( parameters_.eta_!=0 ){
    eta_CAP_correct(rearranged_E_points, partial_result);
    eta_CAP_correct(rearranged_E_points, rvec_partial_result);
  }
  */


  /*-----------------
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
    conv_R[ 2 * ( r - 1 ) ] = max;
    conv_R[ 2 * ( r - 1 ) + 1 ] = av;
  }
  */
  
  //prev_partial_result_=partial_result;


  
  std::ofstream dataR;
  dataR.open(run_dir+"vecs/r"+std::to_string(r)+"_"+filename);

  for(int e=0;e<nump;e++)  
    dataR<< a * rearranged_E_points[e] - b<<"  "<< rvec_partial_result [e]<<"  "<< partial_result [e]  <<std::endl;

  dataR.close();
  


  
  std::ofstream dataP;
  dataP.open(run_dir+"currentResult_"+filename);

  for(int e=0;e<nump;e++)  
    dataP<< a * rearranged_E_points[e] - b<<"  "<<  partial_result [e] <<std::endl;

  dataP.close();



  
  std::ofstream data;
  data.open(run_dir+"conv_R_"+filename);

  for(int r=1;r<D*R;r++)  
    data<< r <<"  "<< conv_R[ 2*(r-1) ]<<"  "<< conv_R[ 2*(r-1) + 1 ] <<std::endl;

  data.close();  

  
  
  
  std::ofstream data2;
  data2.open( run_dir + filename + "_integrand");

  for(int e=0;e<nump;e++)  
    data2<< a * rearranged_E_points[e] - b<<"  "<<  integrand[e] <<std::endl;
  
  data2.close();


  plot_data(run_dir,filename);
}



void Kubo_solver_filtered::update_data_Sea(r_type E_points[], type r_data[], type final_data[], r_type conv_R[], int r, std::string run_dir, std::string filename){

  const std::complex<double> im(0,1);  
  
  
  int nump = parameters_.num_p_,
    R = parameters_.R_,
    D = parameters_.dis_real_;
  
  int SUBDIM = device_.parameters().SUBDIM_;    

  r_type a = parameters_.a_,
    b = parameters_.b_,
    sysSubLength = device_.sysSubLength();
  
  int decRate = filter_.parameters().decRate_;   
  
  r_type omega = 2.0 * decRate * decRate * SUBDIM/( a * a * sysSubLength * sysSubLength ) /* ( 2 * M_PI )*/;//Dimensional and normalizing constant. The minus is due to the vel. op being conjugated.
  //DIM/( a * a );//* sysSubLength * sysSubLength );//Dimensional and normalizing constant
  
  //r_value_t tmp, max=0, av=0;

  r_type
    rearranged_E_points[nump],
    integrand[nump],
    rvec_integrand[nump],
    partial_result[nump],
    rvec_partial_result[nump];

  for(int e=0; e<nump; e++){
    rearranged_E_points[e] = E_points[e];
    integrand[e]           = 0.0;
    rvec_integrand[e]      = 0.0;
    partial_result[e]      = 0.0;
    rvec_partial_result[e] = 0.0;
  }   

  
    


  
  for( int e = 0; e < 2*nump; e++ ){  
    //prev_partial_result[e] = final_data[e];
    final_data[e] = ( final_data [e] * (r-1.0) +  r_data[e] ) / r;
  }


  

    

  //Keeping just the real part of E*p(E)+im*sqrt(1-E^2)*w(E) yields the Kubo-Bastin integrand:
  for(int k = 0; k < nump; k++){
    
    integrand[k]  = -E_points[k] * imag( final_data[ k ] ) - ( sqrt(1.0 - E_points[ k ] * E_points[ k ] ) * imag( final_data[ k + nump ] ) );
    integrand[k] *= 1.0 / pow( (1.0 - E_points[k]  * E_points[k] ), 2.0);
    integrand[k] *=  omega ; 

    rvec_integrand[k]  = -E_points[k] * imag( r_data[ k ] ) - ( sqrt(1.0 - E_points[ k ] * E_points[ k ] ) * imag( r_data[ k + nump ] ) );
    rvec_integrand[k] *= 1.0 / pow( (1.0 - E_points[k]  * E_points[k] ), 2.0);
    rvec_integrand[k] *=  omega ; 
  }

  rearrange_crescent_order(rearranged_E_points);
  rearrange_crescent_order(integrand);
  rearrange_crescent_order(rvec_integrand);

  


  time_station time_integration;
      
  integration_linqt(rearranged_E_points, rvec_integrand, rvec_partial_result);  
  /*integration_linqt*/ interpolated_integration(rearranged_E_points, integrand, partial_result);    

  time_integration.stop("       Integration time:           ");

  

  /*
  if( parameters_.eta_!=0 ){
    eta_CAP_correct(rearranged_E_points, partial_result);
    eta_CAP_correct(rearranged_E_points, rvec_partial_result);
  }
  */


  /*-----------------
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
    conv_R[ 2 * ( r - 1 ) ] = max;
    conv_R[ 2 * ( r - 1 ) + 1 ] = av;
  }
  */
  
  //prev_partial_result_=partial_result;


  
  std::ofstream dataR;
  dataR.open(run_dir+"vecs/r"+std::to_string(r)+"_"+filename);

  for(int e=0;e<nump;e++)  
    dataR<< a * rearranged_E_points[e] - b<<"  "<< rvec_partial_result [e]<<"  "<< partial_result [e]  <<std::endl;

  dataR.close();
  


  
  std::ofstream dataP;
  dataP.open(run_dir+"currentResult_"+filename);

  for(int e=0;e<nump;e++)  
    dataP<< a * rearranged_E_points[e] - b<<"  "<<  partial_result [e] <<std::endl;

  dataP.close();



  
  std::ofstream data;
  data.open(run_dir+"conv_R_"+filename);

  for(int r=1;r<D*R;r++)  
    data<< r <<"  "<< conv_R[ 2*(r-1) ]<<"  "<< conv_R[ 2*(r-1) + 1 ] <<std::endl;

  data.close();  

  
  
  
  std::ofstream data2;
  data2.open( run_dir + filename + "_integrand");

  for(int e=0;e<nump;e++)  
    data2<< a * rearranged_E_points[e] - b<<"  "<<  integrand[e] <<std::endl;
  
  data2.close();


  plot_data(run_dir,filename);
}




void Kubo_solver_filtered::update_data(r_type E_points[],  type r_data[], type final_data[], r_type conv_R[], int r, std::string run_dir, std::string filename){

  int nump = parameters_.num_p_,
      R = parameters_.R_,
      D = parameters_.dis_real_;


  int SUBDIM = device_.parameters().SUBDIM_;    

  r_type a = parameters_.a_,
    b = parameters_.b_,
    sysSubLength = device_.sysSubLength();


  
  r_type omega = SUBDIM/( a * a * sysSubLength * sysSubLength ) /* ( 2 * M_PI )*/ ;//Dimensional and normalizing constant
  
  r_type tmp, max=0, av=0;

  
  //Post-processing     
  int decRate = filter_.parameters().decRate_;   
  for(int m=0;m<nump;m++)
    r_data[m] *= 2.0 * decRate * decRate  / (  1.0 - E_points[m] * E_points[m] );



  
  for(int e=0;e<nump;e++){

    tmp = real (final_data[e]);
    final_data[e] = ( final_data [e] * (r-1.0) + omega * r_data[e] ) / r;

    if(r>1){

      tmp = std::abs( ( final_data [e] - tmp ) / tmp) ;
      if(tmp>max)
        max = tmp;

      av += tmp / nump ;
    }
  }

  if(r>1){
    conv_R[ 2 * (r-1) ]   = max;
    conv_R[ 2 * (r-1)+1 ] = av;
  }

  std::ofstream dataR;
  dataR.open(run_dir+"vecs/r"+std::to_string(r)+"_"+filename);

  
  for(int e=0;e<nump;e++)  
    dataR<< a * E_points[e] - b<<"  "<< real( omega * r_data [e] )<<"  "<< real(final_data [e]) <<std::endl;
    //  dataP<<  e <<"  "<< final_data [e] <<std::endl;

  
  dataR.close();



  
  std::ofstream dataP;
  dataP.open(run_dir+"currentResult_"+filename);

    for(int e=0;e<nump;e++){    
      dataP<< a * E_points[e] - b<<"  "<< real(final_data [e]);

      if(e<nump/2)
        dataP<<"  "<<e;
      else
        dataP<<"  "<<-(nump-e);

      dataP<<"  "<<e<<std::endl;
    }

  dataP.close();


  
  
  std::ofstream data;
  data.open(run_dir+"conv_R_"+filename);

  for(int r=1;r<D*R;r++)  
    data<< r <<"  "<< conv_R[ 2*(r-1) ]<<"  "<< conv_R[ 2*(r-1) + 1 ] <<std::endl;

  data.close();


}






void Kubo_solver_filtered::plot_data(std::string run_dir, std::string filename){
        //VIEW commands

     std::string exestring=
         "gnuplot<<EOF                                               \n"
         "set encoding utf8                                          \n"
         "set terminal pngcairo enhanced                             \n"

         "unset key  \n"

         "set output '"+run_dir+filename+".png'                \n"

         "set xlabel 'E[eV]'                                               \n"
         "set ylabel  'G [2e^2/h]'                                           \n"
         
        "plot '"+run_dir+"currentResult_"+filename+"' using 1:2 w p ls 7 ps 0.25 lc 2;  \n"
         "EOF";
     
      char exeChar[exestring.size() + 1];
      strcpy(exeChar, exestring.c_str());    
      if(system(exeChar)){};


}










void Kubo_solver_filtered::eta_CAP_correct(r_type E_points[], r_type r_data[]){
  int nump = parameters_.num_p_;
  
  for(int e=0;e<nump;e++)
    r_data[e] *= sin(acos(E_points[e]));
}


void Kubo_solver_filtered::integration(r_type E_points[], r_type integrand[], r_type data[]){

  int M = parameters_.M_;
  r_type edge = parameters_.edge_;
  
#pragma omp parallel for 
  for(int k=0; k<M-int(M*edge/4.0); k++ ){  //At the very edges of the energy plot the weight function diverges, hence integration should start a little after and a little before the edge.
                                       //The safety factor guarantees that the conductivity is zero way before reaching the edge. In fact, the safety factor should be used to determine
                                      //the number of points to be ignored in the future;
    for(int j=k; j<M-int(M*edge/4.0); j++ ){//IMPLICIT PARTITION FUNCTION. Energies are decrescent with e (bottom of the band structure is at e=M);
      r_type ej  = E_points[j],
	ej1      = E_points[j+1],
	de       = ej-ej1,
        integ    = ( integrand[j+1] + integrand[j] ) / 2.0;     
      
      data[k] +=  de * integ;
    }
  }
}
