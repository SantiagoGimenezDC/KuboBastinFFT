#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>


#include "fftw_wrapper.hpp"
#include "Kubo_solver_filtered.hpp"
    

void Kubo_solver_filtered::Greenwood_FFTs(std::complex<r_type>** bras, std::complex<r_type>** kets, r_type r_data[]){  

  int nump    = parameters_.num_p_,
      size    = parameters_.SECTION_SIZE_,
      M_dec   = filter_.M_dec();
  
  const std::complex<double> im(0,1);  

  
#pragma omp parallel 
  {
    int id,  Nthrds, l_start, l_end;
    id      = omp_get_thread_num();
    Nthrds  = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end   = (id+1) * size / Nthrds;

    
    if (id == Nthrds-1)
      l_end = size;

    
    
    r_type thread_data [nump];
    for(int m = 0; m < nump; m++)
      thread_data[ m ] = 0;

    
    
    out_of_place_dft
      re_bras( nump, BACKWARD ),
      re_kets( nump, BACKWARD ),
      im_bras( nump, BACKWARD ),
      im_kets( nump, BACKWARD );
                     

    
    # pragma omp critical
    {
      re_bras.create();
      re_kets.create();
      im_bras.create();
      im_kets.create();
    } 

    //SO the solution here is to store separetely the two parts of each vector: real and imaginary. 
        
    for(int l = l_start; l < l_end;l++){      
      for(int m = 0; m < M_dec; m++){
	re_bras.input()[ m ] = real( bras[ m ][ l ] ) ;
        re_kets.input()[ m ] = real( kets[ m ][ l ] ) ;
	
	im_bras.input()[ m ] = imag( bras[ m ][ l ] ) ;
        im_kets.input()[ m ] = imag( kets[ m ][ l ] ) ;	
      }
      
      re_bras.execute();
      re_kets.execute();
      
      im_bras.execute();
      im_kets.execute();
      
      for(int m = 0; m < nump; m++ )
        thread_data[ m ] += real(
	  ( real( re_bras( m ) ) + im * real( im_bras( m ) ) ) *
	  ( real( re_kets( m ) ) + im * real( im_kets( m ) ) )
				 );
			
    }


    # pragma omp critical
    {
      for(int m=0;m<nump;m++)
	r_data[m] +=  thread_data[m];
    }
  }
}




