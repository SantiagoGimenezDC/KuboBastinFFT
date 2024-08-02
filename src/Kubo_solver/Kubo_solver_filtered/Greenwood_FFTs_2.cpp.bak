#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>


#include "fftw_wrapper.hpp"
#include "Kubo_solver_filtered.hpp"
    

void Kubo_solver_filtered::Greenwood_FFTs_2(std::complex<r_type>** bras, std::complex<r_type>** kets, r_type r_data[]){  

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
      bra( nump, BACKWARD ),
      ket( nump, BACKWARD ),
      conj_bra( nump, BACKWARD ),
      conj_ket( nump, BACKWARD );
                     

    
    # pragma omp critical
    {
      bra.create();
      ket.create();
      conj_bra.create();
      conj_ket.create();
    } 

    //SO the solution here is to store separetely the two parts of each vector: real and imaginary. 
        
    for(int l = l_start; l < l_end;l++){      
      for(int m = 0; m < M_dec; m++){
	bra.input()[ m ] = bras[ m ][ l ];
        ket.input()[ m ] = kets[ m ][ l ];
	
	conj_bra.input()[ m ] = conj( bras[ m ][ l ] ) ;
        conj_ket.input()[ m ] = conj( kets[ m ][ l ] ) ;	
      }
      
      bra.execute();
      ket.execute();
      
      conj_bra.execute();
      conj_ket.execute();
      
      for(int m = 0; m < nump; m++ )
        thread_data[ m ] += real( bra(m) - conj(conj_bra(m)) ) * real( ket(m) - conj( conj_ket(m) ) )/2;
      /*
	  real(
	  ( real( re_bras( m ) ) - im * real( im_bras( m ) ) ) *
	  ( real( re_kets( m ) ) + im * real( im_kets( m ) ) )
	  );*/
			
    }


    # pragma omp critical
    {
      for(int m=0;m<nump;m++)
	r_data[m] +=  thread_data[m];
    }
  }
}




