#include "TBG.hpp"
#include<chrono>
#include<iostream>
#include<fstream>
#include<vector>





TBG::TBG(device_vars& device_data) : Device(device_data),
				     all_coordinates_(device_data.W_,2*device_data.LE_,2*device_data.C_),
				     top_coordinates_(device_data.W_,device_data.LE_,device_data.C_),
				     bottom_coordinates_(device_data.W_,device_data.LE_,device_data.C_){
    int W      = this->parameters().W_,
        Le     = this->parameters().LE_,
        C      = this->parameters().C_,
        fullLe = (2*C+Le),
        Dim    = 2*fullLe*W,
        subDim = 2*Le*W;

  fullLe_ = fullLe;

  sysLength_   = fullLe/2 * (1.0+sin(M_PI/6))*a0_; 
  sysSubLength_= Le*(1.0+sin(M_PI/6))*a0_;
  
  this->parameters().DIM_         = Dim;
  this->parameters().SUBDIM_      = subDim;


  singleLayerDim_=fullLe*W;


  
  this->setCoordinates();

  all_coordinates_.plotSample();

}




void TBG::bottom_layer_coordinates(){
  int W   = this->parameters().W_,
      fullLe = fullLe_;

  MatrixXp coordinates(3, fullLe*W);
  
  for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n=j*W+i;
      
        coordinates(1,n)=-i*a0_*cos(M_PI/6.0);

      
      if(i%2==1){
          coordinates(0, n)=a0_*(
				    sin(M_PI/6.0) +
	                            (j/2)*(1.0+2.0*sin(M_PI/6.0)) +
				    ((j+1)/2)
				  );	                          	                          
      
      }
      else{
          coordinates(0, n)=a0_*(
	                           ((j+1)/2)*(1.0+2.0*sin(M_PI/6.0)) +
	                           (j/2)
                                  );	                       

      }
    }
  }

  bottom_coordinates_.reset(coordinates);
}

//My device is along the x axis!! j(columns) coordinates correspond to the x variable, i(rows) coordinates to y axis.
void TBG::top_layer_coordinates(){
  int W      = this->parameters().W_,
    fullLe = fullLe_;
  


  MatrixXp coordinates(3,fullLe*W);
  
  for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n=j*W+i;
      
      coordinates(1,n)=-(i+1)*a0_*cos(M_PI/6.0);
      
      if((i+1)%2==1){
        coordinates(0,n) = a0_*(
				      sin(M_PI/6.0) +
	                              (j/2)*(1.0+2.0*sin(M_PI/6.0)) +
				      ((j+1)/2)
				    );
      }
      else{
        coordinates(0,n) = a0_*(
	                             ((j+1)/2)*(1.0+2.0*sin(M_PI/6.0)) +
	                             (j/2)
                                    );
      }
    }
  }

  top_coordinates_.reset(coordinates);

}






void TBG::setCoordinates(){
  int W      = this->parameters().W_,
      Dim    = this->parameters().DIM_,
      fullLe = fullLe_;

  r_type theta  = this->parameters().theta_;


  bottom_layer_coordinates();

  Eigen::Matrix<r_type,3,1> tmpCenter = -bottom_coordinates_.data().col(W*(fullLe/2)+W/2);    
  bottom_coordinates_.centralize();
  

 
  top_layer_coordinates();
  Eigen::Matrix<r_type,3,1> d(0.0,0.0,d0_);
  Eigen::Matrix<r_type,3,1> b(a0_*cos(M_PI/6.0), a0_*sin(M_PI/6.0),0.0);
  
  top_coordinates_.translate( tmpCenter);
  top_coordinates_.translate( d);
  top_coordinates_.translate( b);   
  top_coordinates_.rotate   ( theta);

  MatrixXp  all_coordinates(3, Dim);
  
  
  all_coordinates.block(0,0,3,singleLayerDim_)      = bottom_coordinates_.data();
  all_coordinates.block(0,singleLayerDim_,3,singleLayerDim_) = top_coordinates_.data();

  all_coordinates_.reset(all_coordinates);
  
};














 
void TBG::SlaterCoster_Hamiltonian(SpMatrixXp& H){
  int Dim = this->parameters().DIM_;
  
  std::vector<T> tripletList;
  tripletList.reserve(15*Dim);


  
  
  auto start_RV = std::chrono::steady_clock::now(); 
  
  intralayerNeighbours_SCH(tripletList);
  interlayerNeighbours_SCH(tripletList);


  
  H.resize(Dim,Dim);  
  H.setFromTriplets(tripletList.begin(), tripletList.end(), [] (const r_type &,const r_type &b) { return b; });
  H.makeCompressed();
  H.makeCompressed();

  tripletList.clear();
  
  auto end_RV = std::chrono::steady_clock::now();






  std::cout<<std::endl<<
  std::endl<<"      Number of Hamiltonian entries per atomic site: "<<H.nonZeros()/Dim<<std::endl;
  std::cout<<"       Time to make the FAST TBG Hamiltonian:     ";
  int millisec=std::chrono::duration_cast<std::chrono::milliseconds>
	         (end_RV - start_RV).count();
  int sec=millisec/1000;
  int min=sec/60;
  int reSec=sec%60;
  std::cout<<min<<" min, "<<reSec<<" secs;"<<" ("<< millisec<<"ms) "
	   <<std::endl<<std::endl<<std::endl;
}
  








void TBG::intralayerNeighbours_SCH(std::vector<T> &tripletList){
  int W      = this->parameters().W_,
      Dim    = this->parameters().DIM_,
      Mdef   = 2*this->parameters().d_min_,//2*d_min,
      Mdefi  = 0,
      j      = 0;

  r_type d_min  = this->parameters().d_min_*a0_;
  
  Eigen::Matrix<r_type,3,1> R_ij(0.0,0.0,0.0);

  MatrixXp coordinates = all_coordinates_.data();
  
  r_type d_ij,
         t_ij;




  auto start_RV = std::chrono::steady_clock::now(); 

  
  for(int i=0; i<singleLayerDim_; i++){ 
    for(int j_n=i/W; j_n<=i/W+Mdef && j_n<singleLayerDim_/W; j_n++){
      
      if(j_n == i/W)
	Mdefi=-1;
      else
	Mdefi=Mdef;

      for(int i_n = (i%W-Mdefi>0 ? i%W-Mdefi : 0); i_n<=i%W+Mdef && i_n<W; i_n++){
	
	j    = j_n*W+i_n;
        R_ij = coordinates.col(j)-coordinates.col(i);	    
        d_ij = R_ij.norm();
	
        if(d_ij<d_min){
	      t_ij = SlaterCoster_intralayer_coefficient(d_ij);
              tripletList.push_back(T(i,j,t_ij));
              tripletList.push_back(T(j,i,t_ij));
	}
      }
    }
  }


 
 for(int i=singleLayerDim_; i<Dim; i++){  
    for(int j_n=i/W; j_n<=i/W+Mdef && j_n<Dim/W ;j_n++){
      
      if(j_n == i/W)
	Mdefi=-1;
      else
	Mdefi=Mdef;
	  
      for(int i_n = (i%W-Mdefi>0 ? i%W-Mdefi : 0); i_n<=i%W+Mdef && i_n<W ;i_n++){
	
	j    = j_n*W+i_n;
        R_ij = coordinates.col(j)-coordinates.col(i);	    
        d_ij = R_ij.norm();
      
        if(d_ij<d_min){
	      t_ij = SlaterCoster_intralayer_coefficient(d_ij);
              tripletList.push_back(T(i,j,t_ij));
              tripletList.push_back(T(j,i,t_ij));
	}
      }
    }
  }  
  
  auto end_RV = std::chrono::steady_clock::now();    

  

  std::cout<<"       Time to run through the intralayer elements:     ";
  int millisec=std::chrono::duration_cast<std::chrono::milliseconds>
	         (end_RV - start_RV).count();
  int sec=millisec/1000;
  int min=sec/60;
  int reSec=sec%60;
  std::cout<<min<<" min, "<<reSec<<" secs;"<<" ("<< millisec<<"ms) "
	   <<std::endl<<std::endl<<std::endl;

}






//Again, for equally sized layers;
void TBG::interlayerNeighbours_SCH(std::vector<T> &tripletList){
  int W      = this->parameters().W_,
      Mdef   = 3*this->parameters().d_min_,
      j      = 0,
    cN     = bottom_coordinates_.origin_entries()(0),
    cW     = bottom_coordinates_.origin_entries()(1);

  r_type d_min  = this->parameters().d_min_*a0_,
            theta  = this->parameters().theta_;
  
  MatrixXp coordinates = all_coordinates_.data();




  Eigen::Matrix<r_type,3,1> R_ij(0.0,0.0,0.0),
                               ez(0.0,0.0,1.0),
                               sysSize;

  

  int begin = 0,
      end   = singleLayerDim_;
  
  r_type d_ij,
            t_ij,
            tg_sys;

  //Estimate the ratio between armchair and zigzag axis in Matrix entry coordinates;
  //We use the coordinates of the bottom layer which is aligned with de x and y axis
  sysSize = (coordinates.col(singleLayerDim_-1)-coordinates.col(0));
  tg_sys  = -sysSize(1)/sysSize(0); 
  tg_sys  *= r_type(singleLayerDim_/W)/r_type(W);  

 
  r_type re_theta = -theta;
  MatrixXp  Rotation(2,2);
  Rotation(0,0) = cos(re_theta);
  Rotation(0,1) = -sin(re_theta)/tg_sys;
  Rotation(1,0) = tg_sys*sin(re_theta);
  Rotation(1,1) = cos(re_theta);
  
  Eigen::Matrix<r_type,2,1> B_pos;    
  Eigen::Matrix<r_type,2,1> T_pos;





  

  
  auto start_RV = std::chrono::steady_clock::now();
  
    for(int i=begin;i<end;i++){      
      B_pos=Eigen::Matrix<r_type,2,1>(i%W-cW-1,(i/W-cN-1));      
      T_pos=Rotation*B_pos;
      
      for(int jT=floor(T_pos(1))-Mdef;jT<floor(T_pos(1))+Mdef;jT++){	
        for(int iT=floor(T_pos(0))-Mdef;iT<floor(T_pos(0))+Mdef;iT++){	      
	  if(jT+cN<end/W && jT+cN>=begin/W && iT+cW<W && iT+cW>=0){
	    j=(jT+cN)*W+iT+cW+singleLayerDim_;	
	    R_ij=(coordinates.col(j)-coordinates.col(i));	    
	    d_ij=R_ij.norm();	    
	 
	  
	  if(d_ij<d_min){

	    t_ij = SlaterCoster_coefficient(R_ij, d_ij);
	    
	    tripletList.push_back(T(i,j,t_ij));
	    tripletList.push_back(T(j,i,t_ij));

	  }	    
	}
      }
    }
  }
   
  auto end_RV = std::chrono::steady_clock::now();    






  std::cout<<"       Time to run through the interlayer elements:     ";
  int millisec=std::chrono::duration_cast<std::chrono::milliseconds>
	         (end_RV - start_RV).count();
  int sec=millisec/1000;
  int min=sec/60;
  int reSec=sec%60;
  std::cout<<min<<" min, "<<reSec<<" secs;"<<" ("<< millisec<<"ms) "
	   <<std::endl<<std::endl<<std::endl; 
}







void TBG::naiveHamiltonian(SpMatrixXp& H){
  int Dim    = this->parameters().DIM_;
  
  r_type d_min  = this->parameters().d_min_*a0_;
  
  Eigen::Matrix<r_type,3,1> R_ij(0.0,0.0,0.0), R_ij2(0.0,0.0,0.0),
                  ez(0.0,0.0,1.0),
                  System_Size(0.0,0.0,0.0);
  
  MatrixXp coordinates = all_coordinates_.data();
  r_type d_ij,
            d_ij2,
            t_ij;




  

  
  auto start_RV = std::chrono::steady_clock::now();
  
  std::vector<T> tripletList;
  tripletList.reserve(35*Dim);
  

  for(int i=0;i<Dim;i++){
    for(int j=i+1;j<Dim;j++){	  
      R_ij=(coordinates.col(j)-coordinates.col(i));	      
      d_ij=R_ij.norm();  
      
      if(d_ij<d_min){
	t_ij = VppPI_  * exp(-(d_ij-a0_)/delta_) * (1.0-pow(R_ij.dot(ez)/d_ij,2.0)) +
	       VppSIG_ * exp(-(d_ij-d0_)/delta_) * pow(R_ij.dot(ez)/d_ij,2.0);
	
	tripletList.push_back(T(i,j,t_ij));
	tripletList.push_back(T(j,i,t_ij));
      }
    }
  }
 

   
  H.resize(Dim,Dim);	
  H.setFromTriplets(tripletList.begin(), tripletList.end());
  //Hamiltoniann.prune((double)10e-10);
  H.makeCompressed();

  auto end_RV = std::chrono::steady_clock::now();  



  

  
  std::cout<<std::endl<<std::endl<<"       Number of Hamiltonian entries per atomic site: "<<H.nonZeros()/Dim<<std::endl;
  std::cout<<"       Time to make the TBG Hamiltonian:     ";
  int millisec=std::chrono::duration_cast<std::chrono::milliseconds>
	         (end_RV - start_RV).count();
  int sec=millisec/1000;
  int min=sec/60;
  int reSec=sec%60;
  std::cout<<min<<" min, "<<reSec<<" secs;"<<" ("<< millisec<<"ms) "
	   <<std::endl<<std::endl<<std::endl<<std::endl;
}


void TBG::traceover(type* traced, type* full_vec, int s, int num_reps){
  int subDim = this->parameters().SUBDIM_,
      C   = this->parameters().C_,
      Le  = this->parameters().LE_,
      W   = this->parameters().W_,
      buffer_length = subDim/num_reps;
	
  if( s == num_reps )
      buffer_length =  subDim % num_reps;
  

    
#pragma omp parallel for
  for(int n=0; n<buffer_length;n++){
    int n_full = C*W+s*buffer_length+n;

    if(  s*buffer_length + n < Le * W ){
      traced[n] = full_vec[n_full];
    }
    else {
	traced[n] = full_vec[ ( 3*C+Le ) * W + (s*buffer_length+n) % (Le * W)];
	//	std::cout<<n<<"   "<<( 3*C+Le ) * W + (s*buffer_length+n) % (Le * W)<<"/"<<this->parameters().DIM_-C*W<<std::endl;
   
    }
  }

};


void TBG::adimensionalize ( r_type a, r_type b){
  int Dim = this->parameters().DIM_;
  SpMatrixXp Id(Dim,Dim);
  Id.setIdentity();

  a_=a;
  b_=b;

  H_=(H_+b*Id)/a;
}


void TBG::damp ( r_type damp_op[]){
  int Dim = this->parameters().DIM_,
      C   = this->parameters().C_,
      Le  = this->parameters().LE_,
      W   = this->parameters().W_;
 
  SpMatrixXp Id(Dim,Dim), gamma(Dim,Dim);//, dis(Dim,Dim);  dis.setZero();
  Id.setIdentity();
  gamma = Id;


  #pragma omp parallel for
  for(int i=0; i<Dim;i++)
    gamma.coeffRef(i,i) *=damp_op[ i%(Dim/2) ];

  /*
  #pragma omp parallel for
  for(int i=0; i<Le*W;i++)
     dis.coeffRef(C*W + i, C*W +i)   +=dis_vec[i];
  
  #pragma omp parallel for
  for(int i=Le*W; i<2*Le*W;i++)
     dis.coeffRef(singleLayerDim_ + C*W + i, singleLayerDim_ +C*W +i) += dis_vec[Le*W + i];
  */


  H_ = gamma*(H_);
  
}


void TBG::H_ket ( type* vec, type* p_vec){
  int Dim = this->parameters().DIM_;
  Eigen::Map<VectorXdT> eig_vec(vec,Dim),
    eig_p_vec(p_vec, Dim);

  eig_vec=H_*eig_p_vec;   


}


void TBG::update_cheb ( type vec[], type p_vec[], type pp_vec[], r_type damp_op[], r_type* ){

  int Dim = this->parameters().DIM_;
 
#pragma omp parallel for
  for(int i=0;i<Dim;i++)
    pp_vec[i] *= damp_op[i%(Dim/2)]*damp_op[i%(Dim/2)];
  
  Eigen::Map<VectorXdT> eig_vec(vec,Dim),
    eig_p_vec(p_vec, Dim),
    eig_pp_vec(pp_vec, Dim);

  
  eig_vec = 2.0*H_*eig_p_vec-eig_pp_vec;

  eig_pp_vec = eig_p_vec;
  eig_p_vec = eig_vec;
}


void TBG::vel_op (type vec[], type p_vec[]){
  int Dim = this->parameters().DIM_;
  
  Eigen::Map<VectorXdT> eig_vec(vec,Dim),
    eig_p_vec(p_vec, Dim);

  eig_vec = vx_ * eig_p_vec;
  
};


void TBG::setup_velOp(){
  
  int Dim = this->parameters().DIM_,
      C   = this->parameters().C_,
      Le  = this->parameters().LE_,
      W   = this->parameters().W_;

  vx_.resize(Dim,Dim);
  vx_.setZero();
  
  std::vector<T> tripletList;
  tripletList.reserve(5*Dim);

  MatrixXp coordinates = all_coordinates_.data();


  for (int k=0; k<H_.outerSize(); ++k)
    for (typename SpMatrixXp::InnerIterator it(H_,k); it; ++it)
    {

      int i=it.row(),
	j=it.col();
	//j_corr=0,
	//i_corr=0;

      bool isJ_dev = false,
	   isI_dev = false;
      

      if(j>=C*W  &&  j<(C+Le)*W){
          isJ_dev = true;
	  //j_corr = C*W;	  	  
	}
      else if( j >= (3*C+Le) * W   &&   j < (3*C+2*Le)*W ){
          isJ_dev = true;
	  //j_corr = (3*C+Le) * W;	  	 
      }
      

      if(i>=C*W  &&  i<(C+Le)*W){
          isI_dev = true;
	  //i_corr = C*W;	  	  
       }
      else if( i >= (3*C+Le) * W   &&   i < (3*C+2*Le)*W ){
          isI_dev = true;
	  //	  i_corr = (3*C+Le) * W;	  	 
      }

     
      if(isJ_dev && isI_dev){
	r_type ijHam = it.value(), v_ij;

	v_ij  =  ( coordinates(0,i) - coordinates(0,j) ) * ijHam;
        tripletList.push_back(T(i,j, v_ij) );
      
    }
  }
  vx_.setFromTriplets(tripletList.begin(), tripletList.end(),[] (const r_type &,const r_type &b) { return b; });  


}

