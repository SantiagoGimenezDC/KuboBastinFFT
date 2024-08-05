#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>

#include "Kubo_solver_filtered.hpp"
#include "../fftw_wrapper.hpp"




void Kubo_solver_filtered::Bastin_FFTs_doubleBuffer_imag  (r_type E_points[],
							   std::complex<r_type>** bras_re, std::complex<r_type>** bras_im,
							   std::complex<r_type>** d_bras_re, std::complex<r_type>** d_bras_im,
							   std::complex<r_type>** kets_re, std::complex<r_type>** kets_im,
							    std::complex<r_type>** d_kets_re, std::complex<r_type>** d_kets_im,
							   type* r_data, int s){

  const std::complex<double> im(0,1);
  
  int M    = parameters_.M_,
      size = parameters_.SECTION_SIZE_,
       nump = parameters_.num_p_,
       num_parts = parameters_.num_parts_;
  
  int M_dec = filter_.M_dec(),
      M_ext = filter_.parameters().M_ext_,
      L = filter_.parameters().L_,
      Np = (L-1)/2;

  std::vector<int> list = filter_.decimated_list(),
    sign(nump);  


  for(int i =0; i<nump;i++){
    double E_prev ;

    if( i == 0 && nump == M_ext )
       E_prev = E_points[ nump - 1 ];
    else
         E_prev = E_points[ i - 1 ];
    
    if( E_points[i] - E_prev < 0 )
      sign[i] = 1;
    else
      sign[i] = -1;
  }

  
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
      bras_re_dft( nump, BACKWARD ),
      kets_re_dft( nump, BACKWARD ),
      bras_im_dft( nump, BACKWARD ),
      kets_im_dft( nump, BACKWARD ),

      d_bras_re_dft( nump, BACKWARD ),
      d_kets_re_dft( nump, BACKWARD ),
      d_bras_im_dft( nump, BACKWARD ),
      d_kets_im_dft( nump, BACKWARD );



   std::complex<double> p[nump], w[nump]; //Dot product partial results;

   std::vector<type>  D_bras_re(nump), D_bras_im(nump), D_kets_re(nump), D_kets_im(nump); 

   
   for(int k = 0; k < nump; k++){
     p[k] = 0;
     w[k] = 0;
   }
    

    //FFTW plans. All plans are in-place backward 1D FFTS of the entry variables; These functions are no thread safe for some reason, hence the # pragma critical
# pragma omp critical
   {
      bras_re_dft.create();
      kets_re_dft.create();
      bras_im_dft.create();
      kets_im_dft.create();

      d_bras_re_dft.create();
      d_kets_re_dft.create();
      d_bras_im_dft.create();
      d_kets_im_dft.create();

   }

   for(int l = l_start; l < l_end; l++){

      if( M_ext > M + Np ){
        for(int m=0; list[m] < M + Np; m++){
	  bras_re_dft.input()[ m ] = bras_re[ m ][ l ];
	  kets_re_dft.input()[ m ] = kets_re[ m ][ l ];
	  bras_im_dft.input()[ m ] = bras_im[ m ][ l ];
	  kets_im_dft.input()[ m ] = kets_im[ m ][ l ];         

          d_bras_re_dft.input()[ m ] = d_bras_re[ m ][ l ];
	  d_kets_re_dft.input()[ m ] = d_kets_re[ m ][ l ];
	  d_bras_im_dft.input()[ m ] = d_bras_im[ m ][ l ];
	  d_kets_im_dft.input()[ m ] = d_kets_im[ m ][ l ];         
        
	}
        for(int m=0; list[M_dec - 1 - m ] > M_ext - 1 - Np; m++ ){
	  bras_re_dft.input()[ nump - 1 - m ] = bras_re[ M_dec - 1 - m ][ l ];
	  kets_re_dft.input()[ nump - 1 - m ] = kets_re[ M_dec - 1 - m ][ l ];
	  bras_im_dft.input()[ nump - 1 - m ] = bras_im[ M_dec - 1 - m ][ l ];
	  kets_im_dft.input()[ nump - 1 - m ] = kets_im[ M_dec - 1 - m ][ l ];         

	  d_bras_re_dft.input()[ nump - 1 - m ] = d_bras_re[ M_dec - 1 - m ][ l ];
	  d_kets_re_dft.input()[ nump - 1 - m ] = d_kets_re[ M_dec - 1 - m ][ l ];
	  d_bras_im_dft.input()[ nump - 1 - m ] = d_bras_im[ M_dec - 1 - m ][ l ];
	  d_kets_im_dft.input()[ nump - 1 - m ] = d_kets_im[ M_dec - 1 - m ][ l ];         

	}
      }
      else	
        for(int m = 0; m < nump; m++){
	  bras_re_dft.input()[ m ] = bras_re[ m ][ l ];
	  kets_re_dft.input()[ m ] = kets_re[ m ][ l ];
	  bras_im_dft.input()[ m ] = bras_im[ m ][ l ];
	  kets_im_dft.input()[ m ] = kets_im[ m ][ l ];         

	  d_bras_re_dft.input()[ m ] = d_bras_re[ m ][ l ];
	  d_kets_re_dft.input()[ m ] = d_kets_re[ m ][ l ];
	  d_bras_im_dft.input()[ m ] = d_bras_im[ m ][ l ];
	  d_kets_im_dft.input()[ m ] = d_kets_im[ m ][ l ];        
	}


      bras_re_dft.execute();
      kets_re_dft.execute();
      
      bras_im_dft.execute();
      kets_im_dft.execute();


      d_bras_re_dft.execute();
      d_kets_re_dft.execute();
      
      d_bras_im_dft.execute();
      d_kets_im_dft.execute();
      
      /*      
      derivate_4th_order_imag(D_bras_re, bras_re_dft,  nump, M_ext);
      derivate_4th_order_imag(D_bras_im, bras_im_dft,  nump, M_ext);

      derivate_4th_order_imag(D_kets_re, kets_re_dft,  nump, M_ext);      
      derivate_4th_order_imag(D_kets_im, kets_im_dft,  nump, M_ext);
      */


     for(int j = 0; j < nump; j++){
       if( sign[j] == 1 ){
       //Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
       p[j] += ( real( bras_re_dft(j) ) - im * real( bras_im_dft(j) ) ) *  //Re(G(k)) *
	         (       kets_re_dft(j)   + im *       kets_im_dft(j) ) + //G(k)+
 	  
	         ( conj( bras_re_dft(j) ) - im * conj( bras_im_dft(j) ) ) * //G(k)
	         ( real( kets_re_dft(j) ) + im * real( kets_im_dft(j) ) );   //Re(G(k))
	

       //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
       w[j] += ( conj( d_bras_re_dft(j) ) - im * conj( d_bras_im_dft(j) ) )  * //dG(k)
		 ( real( kets_re_dft(j)   ) + im * real( kets_im_dft(j) ) ) - //Re(G(k))
	  
		 ( real( bras_re_dft(j) )   - im * real( bras_im_dft(j) ) )  *  //Re(G(k))
	         ( d_kets_re_dft(j)         + im * d_kets_im_dft(j) ); //dG(k)
       }

       if( sign[j] == -1 ){
       //Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
       p[j] += ( real( bras_re_dft(j) ) - im * real( bras_im_dft(j) ) ) *  //Re(G(k)) *
	       ( conj( kets_re_dft(j) ) + im * conj( kets_im_dft(j) ) ) + //G(k)+
 	  
	         ( bras_re_dft(j)   - im * bras_im_dft(j)  ) * //G(k)
	         ( real( kets_re_dft(j) ) + im * real( kets_im_dft(j) ) );   //Re(G(k))
	

         //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
         w[j] += (  d_bras_re_dft(j)  - im *  d_bras_im_dft(j)  )  * //dG(k)
		 ( real( kets_re_dft(j)   ) + im * real( kets_im_dft(j) ) ) - //Re(G(k))
	  
		 ( real( bras_re_dft(j) )   - im * real( bras_im_dft(j) ) )  *  //Re(G(k))
	         ( conj( d_kets_re_dft(j) )         + im * conj ( d_kets_im_dft(j) ) ); //dG(k)

       }
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
