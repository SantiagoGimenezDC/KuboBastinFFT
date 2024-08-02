#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>

#include "Kubo_solver_filtered.hpp"
#include "../fftw_wrapper.hpp"

/*
inline type derivate_2nd_order_2(  out_of_place_dft& series, int j, int nump,int M_ext, std::vector<int> sign){
  type der=0;

  if( j == 0 || j == nump - 1 )
    der=0;
  else{    

    //      if( j < nump - 1 ){
      type jm1_v = series(j+1),
	   j_v = series(j),
           jp1_v = series(j-1);

      if( sign[j+1] == -1 ) jm1_v = conj(jm1_v);      
      if( sign[j] == -1 ) j_v = conj(j_v);
      if( sign[j-1] == -1 ) jp1_v = conj(jp1_v);


      //term1 = jm1_v * ( (x1 - x2) / ((x0 - x1) * (x0 - x2)) );
      //term2 = j_v * (2 * x1 - x0 - x2) / ((x1 - x0) * (x1 - x2));
      //term3 = jp1_v * (x1 - x0) / ((x2 - x0) * (x2 - x1));
      
      der = (jp1_v-jm1_v)/2*(M_ext/(type(0,1)*2.0*M_PI));//type(0,-1)*(term1 + term2 + term3);
      //}
  }

  return der; 

} */

inline void derivate_4th_order_2(std::vector<type>& derivative, out_of_place_dft& series,  int nump, int M_ext){
  type jm2_v, jm1_v, j_v, jp1_v, jp2_v; 
   
  for(int j=0; j< nump;j++){ //running though all j's this way  implies a cyclic BC at num/2 and nump/2+1, which shouldnt be a problem 
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
 
      derivative[j] = ( - jp2_v + 8 * jp1_v - 8 * jm1_v + jm2_v ) / 12 * ( M_ext / ( type(0,1) * 2.0 * M_PI));
    
  }
  } 

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

  
  std::ofstream dataP;
  dataP.open("derivative_data.dat");
   


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
   std::vector<type> D_bras(nump), D_kets(nump); //Dot product partial results;

   
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

     derivate_4th_order_2(D_bras, bras_dft,  nump, M_ext);
     derivate_4th_order_2(D_kets, kets_dft,  nump, M_ext);
     
          
     for(int j = 0; j < nump; j++){

       if( sign[j] == 1 ){

	 /*
	if(l == size/2+size/6-15){
	  //	  dataP<<j<<" "<< real(bras_dft(j))<<" "<< imag(bras_dft(j))<<"     "<<real(conj(D_bras[j]))<<" "<< imag(conj(D_bras[j]))<<"      "<<real(conj( D_bras_dft(j) ))<<" "<< imag(conj( D_bras_dft(j) ))<<std::endl;
	  dataP<<j<<" "<< real(kets_dft(j))<<" "<< imag(kets_dft(j))<<"     "<<real(conj(D_kets[j]))<<" "<< imag(conj(D_kets[j]))<<"      "<<real(conj( D_kets_dft(j) ))<<" "<< imag(conj( D_kets_dft(j) ))<<std::endl;	  
	  //dataP<<j<<" "<< kets_dft(j)<<"     "<<D_kets<<"  "<< D_kets_dft(j) <<std::endl;
	  }*/

	//Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
         p[j] += 
	   real( bras_dft(j) ) * ( kets_dft(j)   ) + //Re(G(k)) * G(k)+
	   conj( bras_dft(j) ) * real( kets_dft(j) );   //G(k) * Re(G(k))
	

         //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
         w[j] +=  conj( D_bras_dft(j) )  *  real( kets_dft(j) ) - //dG(k) * Re(G(k))-	  
	   real( bras_dft(j) )  *  D_kets_dft(j) ; //Re(G(k)) * dG(k)
       }

       
       if( sign[j] == -1 ){
	 /*
	if(l == size/2+size/6-15){
	  //	  dataP<<j<<" "<< real(bras_dft(j))<<" "<< imag(bras_dft(j))<<"     "<<real(D_bras[j])<<" "<< imag(D_bras[j])<<"      "<<real( D_bras_dft(j) )<<" "<< imag( D_bras_dft(j) )<<std::endl;
	  dataP<<j<<" "<< real(kets_dft(j))<<" "<< imag(kets_dft(j))<<"     "<<real(conj(D_kets[j]))<<" "<< imag(conj(D_kets[j]))<<"      "<<real(conj( D_kets_dft(j) ))<<" "<< imag(conj( D_kets_dft(j) ))<<std::endl;	  
	  //dataP<<j<<" "<< kets_dft(j)<<"     "<<D_kets<<"  "<< D_kets_dft(j) <<std::endl;
	  }*/


	 
       //Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
         p[j] += ( real( bras_dft(j) )  ) * ( conj(kets_dft(j) )   ) + // //Re(G(k)) *G(k)+
	         ( bras_dft(j)     ) * ( real( kets_dft(j) )  );   ////G(k)Re(G(k))
	

         //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
         w[j] += (  D_bras_dft(j)   )  * ( real( kets_dft(j)   )  ) - ////dG(k)Re(G(k))
	   ( real( bras_dft(j) )    )  * ( conj( D_kets_dft(j) )  ); // //Re(G(k))dG(k)

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
  dataP.close();
}
