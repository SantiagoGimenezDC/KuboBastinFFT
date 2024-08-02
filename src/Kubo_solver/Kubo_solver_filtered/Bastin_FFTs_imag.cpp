#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>

#include "Kubo_solver_filtered.hpp"
#include "../fftw_wrapper.hpp"



inline void derivate_4th_order_imag(std::vector<type>& derivative, out_of_place_dft& series,  int nump, int M_ext){
  type jm2_v, jm1_v, j_v, jp1_v, jp2_v; 
   
  for(int j=0; j< nump;j++){ //running through all j's this way  implies a cyclic BC at num/2 and nump/2+1, which shouldnt be a problem 
    jm2_v = series(j-2),
    jm1_v = series(j-1),
    j_v = series(j),
    jp1_v = series(j+1),
    jp2_v = series(j+2);
      
    if ( j == 0 ){
      jm2_v = series(nump - 2);
      jm1_v = series(nump - 1);
    }
    if ( j == 1 ){
      jm2_v = series(nump - 1);
    }
    if ( j == nump - 1 ){
      jp1_v = series(0),
      jp2_v = series(1);
    }
    if ( j == nump - 2 ){
      jp2_v = series(0);
    }
 
    derivative[j] = ( - jp2_v + 8 * jp1_v - 8 * jm1_v + jm2_v ) / 12 * ( M_ext / ( type(0,1) * 2.0 * M_PI)); //Maybe this is the mistake??
    
  }
} 



void Kubo_solver_filtered::Bastin_FFTs_imag  (r_type E_points[], std::complex<r_type>** bras_re, std::complex<r_type>** bras_im, std::complex<r_type>** kets_re, std::complex<r_type>** kets_im,  type* r_data, int s){

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
      kets_im_dft( nump, BACKWARD );


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
   }

   for(int l = l_start; l < l_end; l++){

      if( M_ext > M + Np ){
        for(int m=0; list[m] < M + Np; m++){
	  bras_re_dft.input()[ m ] = bras_re[ m ][ l ];
	  kets_re_dft.input()[ m ] = kets_re[ m ][ l ];
	  bras_im_dft.input()[ m ] = bras_im[ m ][ l ];
	  kets_im_dft.input()[ m ] = kets_im[ m ][ l ];         
        }
        for(int m=0; list[M_dec - 1 - m ] > M_ext - 1 - Np; m++ ){
	  bras_re_dft.input()[ nump - 1 - m ] = bras_re[ M_dec - 1 - m ][ l ];
	  kets_re_dft.input()[ nump - 1 - m ] = kets_re[ M_dec - 1 - m ][ l ];
	  bras_im_dft.input()[ nump - 1 - m ] = bras_im[ M_dec - 1 - m ][ l ];
	  kets_im_dft.input()[ nump - 1 - m ] = kets_im[ M_dec - 1 - m ][ l ];         
        }
      }
      else	
        for(int m = 0; m < nump; m++){
	  bras_re_dft.input()[ m ] = bras_re[ m ][ l ];
	  kets_re_dft.input()[ m ] = kets_re[ m ][ l ];
	  bras_im_dft.input()[ m ] = bras_im[ m ][ l ];
	  kets_im_dft.input()[ m ] = kets_im[ m ][ l ];         
	}

      
      bras_re_dft.execute();
      kets_re_dft.execute();
      
      bras_im_dft.execute();
      kets_im_dft.execute();

      
      derivate_4th_order_imag(D_bras_re, bras_re_dft,  nump, M_ext);
      derivate_4th_order_imag(D_bras_im, bras_im_dft,  nump, M_ext);

      derivate_4th_order_imag(D_kets_re, kets_re_dft,  nump, M_ext);      
      derivate_4th_order_imag(D_kets_im, kets_im_dft,  nump, M_ext);



     for(int j = 0; j < nump; j++){
       if( sign[j] == 1 ){
       //Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
         p[j] += ( real( bras_re_dft(j) ) - im * real ( bras_im_dft(j) ) ) *
	   ( kets_re_dft(j)         + im * kets_im_dft(j) ) + //Re(G(k)) * G(k)+

	   ( conj( bras_re_dft(j) ) - im * conj(bras_im_dft(j) ) )  *
	   ( real( kets_re_dft(j) ) + im * real( kets_im_dft(j) ) ) ;   //G(k) * Re(G(k))
	

       //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
         w[j] += ( conj( D_bras_re[j] )   - im * conj( D_bras_im[j] ) )    *
	         ( real( kets_re_dft(j) ) + im * real( kets_im_dft(j) ) ) - //dG(k) * Re(G(k))-	  

	         ( real( bras_re_dft(j) ) - im * real( bras_im_dft(j) ) )  *
	         ( D_kets_re[j]           + im * D_kets_im[j] ) ; //Re(G(k)) * dG(k)
       }

       if( sign[j] == -1 ){
       //Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
         p[j] +=  ( real( bras_re_dft(j) ) - im * real( bras_im_dft(j) )) *
	          ( conj(kets_re_dft(j) )   + im * conj(kets_im_dft(j) ) )  + //Re(G(k)) *G(k)+
	   
	          ( bras_re_dft(j)         - im * bras_im_dft(j) )        *
	          ( real( kets_re_dft(j) ) + im * real( kets_im_dft(j) ) ) ;    //G(k)Re(G(k))
	

         //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
         w[j] += ( D_bras_re[j]           - im * D_bras_im[j] )           *
	         ( real( kets_re_dft(j) ) + im * real( kets_im_dft(j) ) ) - //dG(k)Re(G(k))

	         ( real( bras_re_dft(j) ) - im * real( bras_im_dft(j) ) ) *
	         ( conj( D_kets_re[j])    + im * conj( D_kets_im[j] ) ); //Re(G(k))dG(k)

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
