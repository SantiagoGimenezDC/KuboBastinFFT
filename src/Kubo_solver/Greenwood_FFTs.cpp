#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>


#include "fftw_wrapper.hpp"
#include "Kubo_solver_filtered.hpp"
    

void Kubo_solver_filtered::Greenwood_FFTs(std::complex<r_type>** bras, std::complex<r_type>** kets, r_type r_data[]){  

  int nump    = filter_.M_dec(),//parameters_.num_p_,
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
      bras_dft( nump, BACKWARD ),
      kets_dft( nump, BACKWARD );
                     

    
    # pragma omp critical
    {
      bras_dft.create();
      kets_dft.create();
    } 

    //SO the solution here is to store separetely the two parts of each vector: real and imaginary. 
        
    for(int l = l_start; l < l_end;l++){      
      for(int m = 0; m < M_dec; m++){
	bras_dft.input()[ m ] = bras[ m ][ l ]  ;
        kets_dft.input()[ m ] = kets[ m ][ l ]  ;        
      }
      
      bras_dft.execute();
      kets_dft.execute();
      
      bras_dft.execute();
      kets_dft.execute();
      
      for(int m = 0; m < nump; m++ )
        thread_data[ m ] += real( bras_dft( m ) ) * real( kets_dft( m ) );

			
    }


    # pragma omp critical
    {
      for(int m=0;m<nump;m++)
	r_data[m] +=  thread_data[m];
    }
  }
}




