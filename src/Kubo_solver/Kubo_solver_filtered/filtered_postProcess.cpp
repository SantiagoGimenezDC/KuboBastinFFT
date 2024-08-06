
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
  for( int i=0; i < nump-1; i++)
  {
     const double denerg = E_points[i+1]-E_points[i];
     acc +=integrand[i]*denerg;
     result[i] = acc;
  }
}


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
  
  r_type omega = 2.0 * decRate * decRate * SUBDIM/( a * a * sysSubLength * sysSubLength ) / ( 2 * M_PI );//Dimensional and normalizing constant. The minus is due to the vel. op being conjugated.
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
    final_data[e] += ( final_data [e] * (r-1.0) +  r_data[e] ) / r;
  }


  

    

  //Keeping just the real part of E*p(E)+im*sqrt(1-E^2)*w(E) yields the Kubo-Bastin integrand:
  for(int k = 0; k < nump; k++){
    
    integrand[k]  = E_points[k] * real( final_data[ k ] ) - ( sqrt(1.0 - E_points[ k ] * E_points[ k ] ) * imag( final_data[ k + nump ] ) );
    integrand[k] *= 1.0 / pow( (1.0 - E_points[k]  * E_points[k] ), 2.0);
    integrand[k] *=  omega / ( M_PI ); 

    rvec_integrand[k]  = E_points[k] * real( r_data[ k ] ) - ( sqrt(1.0 - E_points[ k ] * E_points[ k ] ) * real( r_data[ k + nump ] ) );
    rvec_integrand[k] *= 1.0 / pow( (1.0 - E_points[k]  * E_points[k] ), 2.0);
    rvec_integrand[k] *=  omega / ( M_PI ); 
  }

  rearrange_crescent_order(rearranged_E_points);
  rearrange_crescent_order(integrand);
  rearrange_crescent_order(rvec_integrand);

  


  time_station time_integration;
      
  integration_linqt(rearranged_E_points, rvec_integrand, rvec_partial_result);  
  integration_linqt(rearranged_E_points, integrand, partial_result);    

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
    dataR<< a * rearranged_E_points[e] - b<<"  "<< rvec_partial_result [e] <<std::endl;

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


  
  r_type omega = SUBDIM/( a * a * sysSubLength * sysSubLength ) / ( 2 * M_PI );//Dimensional and normalizing constant
  
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
    dataR<< a * E_points[e] - b<<"  "<< omega * r_data [e]<<"  "<< real(final_data [e]) <<std::endl;
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
