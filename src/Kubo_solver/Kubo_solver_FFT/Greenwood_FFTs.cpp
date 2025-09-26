#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>


#include "../fftw_wrapper.hpp"
#include "Kubo_solver_FFT.hpp"



void Kubo_solver_FFT::Greenwood_FFTs( storageType bras, storageType kets, std::vector<type>& r_data, int s){

  int M     = parameters_.M_,
      size  = parameters_.SECTION_SIZE_,
      nump = parameters_.num_p_,
      num_parts = parameters_.num_parts_;
  



  if( s != num_parts-1 )
    size -= device_.parameters().SUBDIM_ % num_parts;


 
  type pre_factors [ M ];

  for(int m=0; m<M; m++)
    pre_factors[m]  = ( 2 - ( m == 0 ) ) * kernel_->term(m, M) * std::polar( 1.0, M_PI * m / ( 2.0 * nump ) ) ;
  
  const std::complex<double> im(0,1);

  
#pragma omp parallel 
  {
    int id,  Nthrds, l_start, l_end;
    id      = omp_get_thread_num();
    Nthrds  = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end   = (id+1) * size / Nthrds;
    
    if (id == Nthrds-1) l_end = size;


    
    r_type thread_data [nump];

    for(int k=0;k<nump;k++)
      thread_data[k]=0;

    
    out_of_place_dft
      re_bras( nump, BACKWARD ),
      im_bras( nump, BACKWARD ),
      
      re_kets( nump, BACKWARD ),
      im_kets( nump, BACKWARD );
     
    	
# pragma omp critical
    {
      re_bras.create();
      im_bras.create();
      re_kets.create();
      im_kets.create();
    }

    for(int l=l_start; l<l_end;l++){

      for( int m=0; m<M; m++ ){
	re_bras.input()[m] = pre_factors[m] * real( bras[m][l] );
        im_bras.input()[m] = pre_factors[m] * imag( bras[m][l] );

	re_kets.input()[m] = pre_factors[m] * real( kets[m][l] );
        im_kets.input()[m] = pre_factors[m] * imag( kets[m][l] ); 

        if ((std::isnan(std::real(re_bras.input()[m])) || std::isnan(std::imag(im_bras.input()[m])))) {
	  //std::cout<<"Bras: "<<l<<"  "<<m<<"  "<<std::real(re_bras.input()[m])<<std::endl;
	  re_bras.input()[m]=0.0;
	  im_bras.input()[m]=0.0;

        }

        if ((std::isnan(std::real(re_kets.input()[m])) || std::isnan(std::imag(im_kets.input()[m])))) {
	  //std::cout<<"Kets: "<<l<<"  "<<m<<"  "<<std::real(re_kets.input()[m])<<std::endl;
	  re_kets.input()[m]=0.0;
	  im_kets.input()[m]=0.0;

        }

      }

      
      re_bras.execute();
      im_bras.execute();
      re_kets.execute();
      im_kets.execute();

      
      for(int k=0; k<nump; k++ ){
      /*        
if ((std::isnan(std::real(re_bras(k))) || std::isnan(std::imag(im_bras(k))))) {
	  std::cout<<"Result Bras: "<<l<<"  "<<k<<"  "<<std::real(re_bras(k))<<std::endl;
	 
        }

        else if ((std::isnan(std::real(re_kets(k))) || std::isnan(std::imag(im_kets(k))))) {
	  std::cout<<"Result Kets: "<<l<<"  "<<k<<"  "<<std::real(re_kets(k))<<std::endl;
	
        }

	else*/
	thread_data[k] += real (
				 ( real( re_bras(k) ) - im * real( im_bras(k) ) ) *
				 ( real( re_kets(k) ) + im * real( im_kets(k) ) )
			       );
      }
      
    }

    # pragma omp critical
    {
      for(int k = 0; k < nump; k++)
	r_data[k] += thread_data[k] ;
    }
  }
}














/*INTERESTING ofr testing
void Kubo_solver::Greenwood_FFTs__imVec_cft(std::complex<r_type> **bras, std::complex<r_type> **kets, r_type r_data[]){

  int M     = parameters_.M_,
      size  = parameters_.SECTION_SIZE_,
      nump = parameters_.num_p_;
 
  type pre_factors [ M ];

  for(int m=0; m<M; m++)
    pre_factors[m]  = kernel_->term(m, M);// ( 2 - ( m == 0 ) ) * kernel_->term(m, M) * std::polar( 1.0, M_PI * m / ( 2.0 * M ) ) ;
  
  const std::complex<double> im(0,1);

  
#pragma omp parallel 
  {
    int id,  Nthrds, l_start, l_end;
    id      = omp_get_thread_num();
    Nthrds  = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end   = (id+1) * size / Nthrds;
    
    if (id == Nthrds-1) l_end = size;


    
    r_type thread_data [nump];

    for(int k=0;k<nump;k++)
      thread_data[k]=0;


      out_of_place_cft
        bras_cft( nump, REDFT01 ),
        kets_cft( nump, REDFT01 );

	
# pragma omp critical
    {
      bras_cft.create();
      kets_cft.create();
    }

    for(int l=l_start; l<l_end;l++){

      


      for( int m = 0; m<M; m++ ){
	bras_cft.re_input()[m] = real( pre_factors[m] ) * real( bras[m][l] );
        bras_cft.im_input()[m] = real( pre_factors[m] ) * imag( bras[m][l] );

	kets_cft.re_input()[m] = real( pre_factors[m] ) * real( kets[m][l] );
        kets_cft.im_input()[m] = real( pre_factors[m] ) * imag( kets[m][l] ); 
      }
      
      
      bras_cft.execute();
      kets_cft.execute();
      

      for(int k=0; k<nump; k++ ){
        thread_data[k] += real(
			       (  bras_cft.re_output()[k] - im * bras_cft.im_output()[k] )  *
			       (  kets_cft.re_output()[k] + im * kets_cft.im_output()[k] ) 
	                  );

      }
    
    }

    # pragma omp critical
    {
      for(int k=0;k<nump;k++)
	r_data[k] += thread_data[k] ;
    }
  }
}
*/
