#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>

#include "Kubo_solver_filtered.hpp"
#include "fftw_wrapper.hpp"



void Kubo_solver_filtered::Bastin_FFTs_doubleBuffer(r_type E_points[], std::complex<r_type>** bras,std::complex<r_type>** d_bras, std::complex<r_type>** kets, std::complex<r_type>** d_kets,  type* r_data, int s){

  const std::complex<double> im(0,1);
  
  int M    = parameters_.M_,
      size = parameters_.SECTION_SIZE_,
       nump = parameters_.num_p_,
       num_parts = parameters_.num_parts_;
  
  int M_dec = filter_.M_dec(),
      M_ext = filter_.parameters().M_ext_,
      L = filter_.parameters().L_,
      Np = (L-1)/2;
  std::vector<int> list = filter_.decimated_list();  



  if( s != num_parts-1)
    size -= device_.parameters().SUBDIM_ % num_parts;

      


#pragma omp parallel 
  {
    int id,  Nthrds, l_start, l_end;
    id      = omp_get_thread_num();
    Nthrds  = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end   = (id+1) * size / Nthrds;

    if (id == Nthrds-1)
      l_end = size;
  
  
  //8 plans + 14 [M] sized vectors per thread. Can it be reduced to 2 plans and some 4 vectors?

    out_of_place_dft
      bras_dft( nump, BACKWARD ),
      kets_dft( nump, BACKWARD ),
      D_bras_dft( nump, BACKWARD ),
      D_kets_dft( nump, BACKWARD );



   std::complex<double> p[nump], w[nump]; //Dot product partial results;

   for(int k = 0; k < nump; k++){
     p[k] = 0;
     w[k] = 0;
   }
    

    //FFTW plans. All plans are in-place backward 1D FFTS of the entry variables; These functions are no thread safe for some reason, hence the # pragma critical
# pragma omp critical
   {
     bras_dft.create();
     kets_dft.create();
     D_bras_dft.create();
     D_kets_dft.create();
   }

   for(int l = l_start; l < l_end; l++){
      if( M_ext > M + Np ){
        int m = 0;
        while( list[m] < M + Np){
	  bras_dft.input()[ m ] = bras[ m ][ l ];
	  kets_dft.input()[ m ] = kets[ m ][ l ];

	  
	  D_bras_dft.input()[ m ] = d_bras[ m ][ l ];
	  D_kets_dft.input()[ m ] = d_kets[ m ][ l ];
          m++;
        }

        m = 0;
        while( list[M_dec - 1 - m ] > M_ext - 1 - Np ){
	  bras_dft.input()[ nump - 1 - m ] = bras[ M_dec - 1 - m ][ l ];
	  kets_dft.input()[ nump - 1 - m ] = kets[ M_dec - 1 - m ][ l ];

	  
	  D_bras_dft.input()[ m ] = d_bras[ m ][ l ];
	  D_kets_dft.input()[ m ] = d_kets[ m ][ l ];
	  
	  m++;
        }
      }
      else	
        for(int m = 0; m < nump; m++){
	  bras_dft.input()[ m ] = bras[ m ][ l ];
	  kets_dft.input()[ m ] = kets[ m ][ l ];
	  D_bras_dft.input()[ m ] = d_bras[ m ][ l ];
	  D_kets_dft.input()[ m ] = d_kets[ m ][ l ];
	}

     bras_dft.execute();
     kets_dft.execute();

     D_bras_dft.execute();
     D_kets_dft.execute();


          
     for(int j = 0; j < nump; j++){
         
       //Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
       p[j] += //real( bras_dft( j ) ) * real( kets_dft( j ) );

	 real( bras_dft(j) ) * ( kets_dft(j)   ) + //Re(G(k)) * G(k)+
	 conj( bras_dft(j) ) * real( kets_dft(j) );   //G(k) * Re(G(k))
	

       //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
       w[j] +=  conj( D_bras_dft(j) )  *  real( kets_dft(j) ) - //dG(k) * Re(G(k))-	  
                real( bras_dft(j) )  *  D_kets_dft(j) ; //Re(G(k)) * dG(k)
		}
           



     
   }


# pragma omp critical
   {
     for(int k = 0; k < nump; k++){
       r_data[ k ]        += p[ k ];
       r_data[ k + nump ] += w[ k ];	
     }
   } 
 }
}
