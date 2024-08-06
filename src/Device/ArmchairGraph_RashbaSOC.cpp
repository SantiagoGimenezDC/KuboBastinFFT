#include "ArmchairGraph_RashbaSOC.hpp"


void ArmchairGraph_RashbaSOC::RashbaSOC_Hamiltonian (SpMatrixXpc& H,  r_type* disorder_potential){

  int fullLe = ( this->parameters().LE_+2*this->parameters().C_ ),
    W      = this->parameters().W_;

  r_type t = t_standard_,
         rashba_str = rashba_str_ * t,
         m_str = t * m_str_;
    

    
  std::complex<r_type> f_y  = 2.0 * rashba_str * cos( M_PI /6.0 ) / 3.0,
                       f_x  = 2.0 * rashba_str / 3.0,
                       f_x2 = 2.0 * rashba_str*sin( M_PI / 6 ) / 3.0;

  //  Eigen::MatrixXi nonContacts = this->nonContactPoints();

  std::complex<r_type> ImUnit(0,1);


  //f_x*=ImUnit;
  //f_x2*=ImUnit;
  f_y*=ImUnit;



  

 
  typedef Eigen::Triplet<std::complex<r_type>> Tc;
  std::vector<Tc> tripletList;
  tripletList.reserve(5*fullLe*W);
  



  //#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;

      tripletList.push_back( Tc( 2*n, 2*n, -m_str ) );
      tripletList.push_back( Tc( (2*n+1), (2*n+1), m_str ) );	
      

      
      if( i!=0 ){
	tripletList.push_back(Tc((2*n),(2*n-2),-t));
	tripletList.push_back(Tc((2*n),(2*n-1),  ( +f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1)  + f_y ) ));

	tripletList.push_back(Tc((2*n+1),(2*n-1),-t));
	tripletList.push_back(Tc((2*n+1),(2*n-2),  ( -f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1)  + f_y ) ));
	
      }
      if(i != (M-1) ){
	tripletList.push_back(Tc((2*n),(2*n+2),-t));
	tripletList.push_back(Tc((2*n),(2*n+3), ( -f_x2 * std::complex<r_type>(((j+i)%2)==0? -1:1) - f_y )  ));

	tripletList.push_back(Tc((2*n+1),(2*n+3),-t));
	tripletList.push_back(Tc((2*n+1),(2*n+2),   ( f_x2 * std::complex<r_type>(((j+i)%2)==0? -1:1) - f_y )    ));

      }
      if(j != (fullLe-1)){
	tripletList.push_back(Tc((2*n),(2*n+2*W),  -(std::complex<r_type>((j+i)%2)) * t ) );
	tripletList.push_back(Tc((2*n),(2*n+2*W+1), (std::complex<r_type>((j+i)%2)) * f_x ) );

	tripletList.push_back(Tc((2*n+1),(2*n+2*W+1), -std::complex<r_type>((j+i)%2) * t ) );
	tripletList.push_back(Tc((2*n+1),(2*n+2*W),  -std::complex<r_type>((j+i)%2) * f_x  ) );

      }

      if(j != 0){
	tripletList.push_back(Tc((2*n),(2*n-2*W),     -std::complex<r_type>((j+i+1)%2) * t));
	tripletList.push_back(Tc((2*n),(2*n-2*W+1),   -std::complex<r_type>((j+i+1)%2) * f_x  ));

	tripletList.push_back(Tc((2*n+1),(2*n-2*W+1), -std::complex<r_type>((j+i+1)%2) * t));
	tripletList.push_back(Tc((2*n+1),(2*n-2*W),    std::complex<r_type>((j+i+1)%2) * f_x  ));
       }
    }
 }  

 /*
 if(disorder_potential!=NULL){
   int current=0;
   for(int c=0; c<nonContacts.rows(); c++){
     for(int n=nonContacts(c,0); n<nonContacts(c,1); n++){
       tripletList.push_back(Tc(n,n,disorder_potential(n-nonContacts(c,0)+current) ));
     }
       current+=nonContacts(c,2);
   }
   }*/

  
 H.resize(fullLe*W, fullLe*W);

  H.setFromTriplets(tripletList.begin(), tripletList.end(),[] (const std::complex<r_type> &a,const std::complex<r_type> &b) { return a+b; });


}; 




void ArmchairGraph_RashbaSOC::update (r_type a, r_type b, type* ket, type* p_ket, type* pp_ket){

  int fullLe = ( this->parameters().LE_+2*this->parameters().C_ ),
      W      = this->parameters().W_;
  
  r_type t = t_standard_/a,
         b_a = b/a,
         m_str = m_str_ * t,
         rashba_str = rashba_str_ * t;
  
  std::complex<r_type> f_y  = 2.0 * rashba_str * cos(M_PI/6.0)/3.0,
                       f_x  = 2.0 * rashba_str / 3.0,
                       f_x2 = 2.0 * rashba_str * sin(M_PI/6.0)/3.0;


  //  Eigen::MatrixXi nonContacts=this->nonContactPoints();
  //Eigen::MatrixXi contactsPts=this->contactPoints();
  //  f_x*=ImUnit;
  // f_x2*=ImUnit;

  std::complex<r_type> ImUnit(0,1);
  f_y *= ImUnit;
 
#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;

      ket[ 2 * n ]     =  2.0 * ( (b_a - m_str ) * p_ket[ 2 * n ] )    - sFilter[ 2 * n ]     * pp_ket[ 2 * n ];
      ket[ 2 * n + 1 ] =  2.0 * ( (b_a + m_str ) * p_ket[ 2 * n + 1] ) - sFilter[ 2 * n + 1 ] * pp_ket[ 2 * n + 1 ];

      
      if( i!=0 ){
	ket[ 2 * n ]     += 2.0 * ( -t * p_ket[ 2 * n - 2 ] + (  f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n - 1 ] );
	ket[ 2 * n + 1 ] += 2.0 * ( -t * p_ket[ 2 * n - 1 ] + ( -f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n - 2 ] );
      }
      if(i != (W-1) ){
	ket[ 2 * n ]     += 2.0 * ( -t * p_ket[ 2 * n + 2 ] + (  f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n + 3 ]   );
	ket[ 2 * n + 1 ] += 2.0 * ( -t * p_ket[ 2 * n + 3 ] + ( -f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n + 2 ]   );
      }
      if(j != (fullLe-1)){
	ket[ 2 * n ]     += 2.0 * ((j+i)%2) * ( -t * p_ket[ 2*n+2*W ]           +  f_x * p_ket[ 2 * n + 2 * W + 1 ] ) ;
	ket[ 2 * n + 1 ] += 2.0 * ((j+i)%2) * ( -t * p_ket[ 2 * n + 2 * W + 1]  -  f_x * p_ket[ 2 * n + 2 * W ]);
      }
      if(j != 0){
	ket[ 2 * n ]     += 2.0 * ((j+i+1)%2) * ( - t * p_ket[ 2 * n - 2 * W ]     -  f_x * p_ket[ 2 * n - 2 * W + 1 ] );
	ket[ 2 * n + 1 ] += 2.0 * ((j+i+1)%2) * ( - t * p_ket[ 2 * n - 2 * W + 1 ] +  f_x * p_ket[ 2 * n - 2 * W ]   );
      }
      
      ket[ 2 * n ]     *= sFilter[ 2 * n ];
      ket[ 2 * n + 1 ] *= sFilter[ 2 * n + 1 ];	       
    }
 }


 if(disorder_potential.size()!=0){
   int current=0;
   for(int c=0; c<nonContacts.rows(); c++){
    #pragma omp parallel for 
     for(int n=nonContacts(c,0); n<nonContacts(c,1); n++){
       ket(n)   +=  2 * sFilter(n) * disorder_potential(n-nonContacts(c,0)+current) * p_ket(n) / a;
     }
       current+=nonContacts(c,2);
   }
 }


 
 bool vertical_cbc=false;
 bool horizontal_cbc=false;


  if(W%2!=0&&vertical_cbc)
   std::cout<<"BEWARE VERTICAL CBC ONLY WORKS FOR EVEN M!!!!"<<std::endl;

  
 if(vertical_cbc)
  for(int j=0; j<fullLe; j++){

    int n_up = j * W + W - 1;
    int n_down = j * W;


    ket[ 2 * n_up ]    += 2.0 * sFilter[ 2 * n_up ]     * ( -t * p_ket[ 2 * n_down ]    + ( -f_x2 * std::complex<r_type>(((j)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n_down + 1] );
    ket[ 2 * n_up + 1] += 2.0 * sFilter[ 2 * n_up + 1 ] * ( -t * p_ket[ 2 * n_down + 1] + (  f_x2 * std::complex<r_type>(((j)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n_down ]  );
      
      
    ket[ 2 * n_down]      += 2.0 * sFilter[ 2 * n_down ] * ( -t * p_ket[ 2 * n_up ] + ( -f_x2 * std::complex<r_type>( ((j+W-1)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n_up + 1 ]   );
    ket[ 2 * n_down + 1 ] += 2.0 * sFilter[ 2 * n_down + 1 ] * ( -t * p_ket[ 2 * n_up + 1 ] + ( f_x2 * std::complex<r_type>(((j+W-1)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n_up ]   );
      
 } 


 if(fullLe%2!=0&&horizontal_cbc)
   std::cout<<"BEWARE HORIZONTAL CBC ONLY WORKS FOR EVEN LE+C!!!!"<<std::endl;

 
 if(horizontal_cbc)
    for(int i=0; i<W; i++){
      int n_front = i;
      int n_back = (fullLe-1) * W + i;

      
	ket[ 2 * n_front ]     += 2.0 * sFilter[ 2 * n_front ]     * ((n_front+1)%2) * ( -t * p_ket[ 2 * n_back ] + ( -f_x * std::complex<r_type>(((n_front)%2)==0? 1:-1) ) * p_ket[ 2 * n_back + 1 ] );
	ket[ 2 * n_front + 1 ] += 2.0 * sFilter[ 2 * n_front + 1 ] * ((n_front+1)%2) *( -t * p_ket(2*n_back+1) + ( f_x * (((n_front)%2)==0? 1:-1)  ) * p_ket[ 2 * n_back ]  );


	ket[ 2 * n_back ]     += 2.0 * sFilter[ 2 * n_back ]     * ((n_front+1)%2) * ( -t * p_ket[ 2 * n_front ] + ( f_x * std::complex<r_type>(((n_back)%2)==0? 1:-1)  ) * p_ket[ 2 * n_front + 1 ]   );
	ket[ 2 * n_back + 1 ] += 2.0 * sFilter[ 2 * n_back + 1 ] * ((n_front+1)%2) * ( -t * p_ket[ 2 * n_front + 1 ] + ( -f_x * std::complex<r_type>(((n_back)%2)==0? 1:-1)  ) * p_ket[ 2 * n_front ]   );

    }


 
}; 




template<typename dataType>
void ArmchairGraphene<dataType>::vx_OTF (r_type coup_str,r_type m_str, r_type a, r_type b, VectorXpc& ket, VectorXpc& p_ket){

  int Le = this->parameters().Le_,
      M      = this->parameters().M_;
  
  r_type t = t_standard_/a;

  if(this->parameters().C_%2==1)
    std::cout<<"BEWARE VX OP ONLY WORKS FOR EVEN C!!!!!!!!!!"<<std::endl;


      
  std::complex<r_type> d_x=a0_,
                          d_x2=a0_*sin(M_PI/6.0),
    f_y = 2.0*coup_str*t*cos(M_PI/6.0)/3.0,
    f_x = 2.0*coup_str*t/3.0,
    f_x2 = 2.0*coup_str*t*sin(M_PI/6.0)/3.0;


  /*

      
      if( i!=0 ){
	ket(2*n)   += 2 * ( -t * p_ket(2*n-2) + ( f_x2 * (((j+i)%2)==0? 1:-1) - f_y ) * p_ket(2*n-1) );
	ket(2*n+1) += 2 * ( -t * p_ket(2*n-1) + ( -f_x2 * (((j+i)%2)==0? 1:-1)  - f_y ) * p_ket(2*n-2)  );
      }
      if(i != (M-1) ){
	ket(2*n)   += 2 * ( -t * p_ket(2*n+2) + ( f_x2 * (((j+i)%2)==0? 1:-1) + f_y ) * p_ket(2*n+3)   );
	ket(2*n+1) += 2 * ( -t * p_ket(2*n+3) + ( -f_x2 * (((j+i)%2)==0? 1:-1) + f_y ) * p_ket(2*n+2)   );
      }
      if(j != (fullLe-1)){
	ket(2*n)   += 2 * ((j+i)%2) * ( -t * p_ket(2*n+2*M)    +  f_x * p_ket(2*n+2*M+1)) ;
	ket(2*n+1) += 2 * ((j+i)%2) * ( -t * p_ket(2*n+2*M+1)  -  f_x * p_ket(2*n+2*M));
      }
      if(j != 0){
	ket(2*n)   += 2 * ((j+i+1)%2) * ( - t * p_ket(2*n-2*M)   -  f_x * p_ket(2*n-2*M+1) );
	ket(2*n+1) += 2 * ((j+i+1)%2) * ( - t * p_ket(2*n-2*M+1) +  f_x * p_ket(2*n-2*M)   );
      }
      

*/

  //  f_x*=ImUnit;
  // f_x2*=ImUnit;

  f_y*=ImUnit;
 
#pragma omp parallel for 
 for(int j=0; j<Le; j++){
    for(int i=0; i<M; i++){
      int n = j * M + i;


        ket(2*n)   = 0;
	ket(2*n+1) = 0;

	
      if( i!=0 ){
	ket(2*n)   += d_x2 * (((j+i)%2)==0? -1:1) * ( -t * p_ket(2*n-2) + ( f_x2 * (((j+i)%2)==0? 1:-1) + f_y ) * p_ket(2*n-1) );
	ket(2*n+1) += d_x2 * (((j+i)%2)==0? -1:1) * ( -t * p_ket(2*n-1) + ( -f_x2 * (((j+i)%2)==0? 1:-1)  + f_y ) * p_ket(2*n-2)  );
      }
      if(i != (M-1) ){
	ket(2*n)   += d_x2 * (((j+i)%2)==0? -1:1) * ( -t * p_ket(2*n+2) + ( f_x2 * (((j+i)%2)==0? 1:-1) - f_y ) * p_ket(2*n+3)   );
	ket(2*n+1) += d_x2 * (((j+i)%2)==0? -1:1) * ( -t * p_ket(2*n+3) + ( -f_x2 * (((j+i)%2)==0? 1:-1) - f_y ) * p_ket(2*n+2)   );
      }
      if(j != (Le-1)){
	ket(2*n)   +=   ((j+i)%2) * d_x * ( t * p_ket(2*n+2*M)    -  f_x * p_ket(2*n+2*M+1)) ;
	ket(2*n+1) +=   ((j+i)%2) * d_x * ( t * p_ket(2*n+2*M+1)  +  f_x * p_ket(2*n+2*M));
      }
      if(j != 0){
	ket(2*n)   +=  - ((j+i+1)%2) * d_x * (  t * p_ket(2*n-2*M)   +  f_x * p_ket(2*n-2*M+1));
	ket(2*n+1) +=  - ((j+i+1)%2) * d_x * (  t * p_ket(2*n-2*M+1) -  f_x * p_ket(2*n-2*M));
      }
      

    }
 } 



 bool vertical_cbc=false;
 bool horizontal_cbc=false;

 if(M%2!=0&&vertical_cbc)
   std::cout<<"BEWARE VERTICAL CBC ONLY WORKS FOR EVEN M!!!!"<<std::endl;
 
 if(vertical_cbc)
  for(int j=0; j<Le; j++){

    int n_up = j * M +M-1;
    int n_down = j * M;


    ket(2*n_up)   +=   d_x2 * (((j)%2)==0? 1:-1) * ( -t * p_ket(2*n_down) + ( -f_x2 * (((j)%2)==0? 1:-1) - f_y ) * p_ket(2*n_down+1) );
    ket(2*n_up+1) +=   d_x2 * (((j)%2)==0? 1:-1) * ( -t * p_ket(2*n_down+1) + ( f_x2 * (((j)%2)==0? 1:-1)  - f_y ) * p_ket(2*n_down)  );
      
      
    ket(2*n_down)   +=  -d_x2 * (((j+M)%2)==0? 1:-1) * ( -t * p_ket(2*n_up) + ( -f_x2 * (((j+M-1)%2)==0? 1:-1) + f_y ) * p_ket(2*n_up+1)   );
    ket(2*n_down+1) +=  -d_x2 * (((j+M)%2)==0? 1:-1) * ( -t * p_ket(2*n_up+1) + ( f_x2 * (((j+M-1)%2)==0? 1:-1) + f_y ) * p_ket(2*n_up)   );
      
 } 


 if(Le%2!=0&&horizontal_cbc)
   std::cout<<"BEWARE HORIZONTAL CBC ONLY WORKS FOR EVEN LE+C!!!!"<<std::endl;
 
 if(horizontal_cbc)
    for(int i=0; i<M; i++){
      int n_front = i;
      int n_back = (Le-1) * M + i;

      
        ket(2*n_front)   +=  - d_x * ((n_front+1)%2) * ( -t * p_ket(2*n_back) + ( f_x * (((n_front)%2)==0? 1:-1)  ) * p_ket(2*n_back+1) );
	ket(2*n_front+1) +=  - d_x * ((n_front+1)%2) * ( -t * p_ket(2*n_back+1) + ( -f_x * (((n_front)%2)==0? 1:-1)  ) * p_ket(2*n_back)  );


	ket(2*n_back)   +=  d_x * ((n_front)%2) * ( -t * p_ket(2*n_front) + ( -f_x * (((n_back)%2)==0? 1:-1) ) * p_ket(2*n_front+1)   );
	ket(2*n_back+1) +=  d_x * ((n_front)%2) * ( -t * p_ket(2*n_front+1) + ( f_x * (((n_back)%2)==0? 1:-1)  ) * p_ket(2*n_front)   );



    }


  std::complex<r_type> im(0,1);
  //   ket*=-im;
 
}; 





