#include<iostream>
#include<fstream>
#include<chrono>
#include "Read_Siesta_Cropped.hpp"
#include<fstream>


#define num_spin_configurations(spin_flag) ((spin_flag) > 2 ? 4 : (spin_flag))
#define num_spin_rows(spin_flag) ((spin_flag) > 1 ? 2 : 1)
#define HEADER_SIZE 5
//typedef size_t indexType




Read_Siesta_Cropped::Read_Siesta_Cropped(device_vars& device_vars):Device(device_vars){};




/** Encodes the ispin multi-index used by SIESTA */
inline int ispin_of_sigmas(const int is1, const int is2) {
    if (is1 == is2) { // diagonal terms
        return is1;
    }
    else {  // off-diagonal
        return 2 + is1;
    }
}





void Read_Siesta_Cropped::create_csr_sparse_matrix(const double prune_threshold,
                                     const int no_u,
                                     const int ncomp,
                                     const int nspin_blocks_buffer,
                                     const int nk,
                                     const cmplx *buffer,
                                     indexType **row_index,
                                     indexType **col_index,
                                     cmplx **values,
                                     indexType *nnz) {

    indexType ispin, ik, is1, is2, io_u, jo_u, icomp, nspin;
    indexType irow, icol;
    bool spin_diagonal;
    cmplx val;

    /** buffer[ (((ik*ncomp + icomp)*nspin_blocks_buffer + ispin)*no_u + jo_u)*no_u + io_u ] 
     *  contains <io_u, k, sigma | O_icomp | jo_u, k, sigma'>
     *  combined row index      = (ik*nsigma + is1)*no_u + io_u
     *  combined column index   = (ik*nsigma + is2)*no_u + jo_u
     *  ispin = ispin_of_sigmas(sigma, sigma')
    */
    // nspin_blocks = num_spin_configurations(nspin_blocks_buffer); already in buffer
    nspin = num_spin_rows(nspin_blocks_buffer);
    spin_diagonal = (nspin_blocks_buffer < 4);
    
    #pragma omp parallel shared(buffer, nnz, row_index, col_index, values),                             \
            firstprivate(ncomp, nk, nspin, no_u, nspin_blocks_buffer, prune_threshold, spin_diagonal),  \
            private(icomp, ik, is1, is2, io_u, jo_u, irow, icol, ispin, val), default(none)
    {

    #pragma omp for schedule(static)
    for (icomp = 0; icomp < ncomp; icomp++) { 
        nnz[icomp] = 0; 
        row_index[icomp][0] = 0;
    }

    /** - rows cannot be parallelized, since nnz depends on previous iterations */
    #pragma omp for schedule(static)
    for (icomp = 0; icomp < ncomp; icomp++) {
        for (ik = 0; ik < nk; ik++) {
            for (is1 = 0; is1 < nspin; is1++) {
                for (io_u = 0; io_u < no_u; io_u++) {
                    /** - outer loops run through rows */
                    irow = (ik*nspin + is1)*no_u + io_u;
                    /** - further loops run through all columns of that row */
                    for (is2 = 0; is2 < nspin; is2++) {
                        if (spin_diagonal && is1 != is2) continue;
                        for (jo_u = 0; jo_u < no_u; jo_u++) {
                            ispin = ispin_of_sigmas(is1, is2);
                            icol = (ik*nspin + is2)*no_u + jo_u;
                            val = buffer[ (((ik*ncomp + icomp)*nspin_blocks_buffer + ispin)*no_u + jo_u)*no_u + io_u ];
                            if (std::abs(val) > prune_threshold) {
                                values[icomp][nnz[icomp]] = val;
                                col_index[icomp][nnz[icomp]] = icol;
                                nnz[icomp]++;
                            }
                        }
                    }
                    /** - row completed; write nnz to row_index array */
                    row_index[icomp][irow+1] = nnz[icomp];
                }
            }
        }
    }

    }   /** - end of parallel region */

    return;
}



int Read_Siesta_Cropped::read_siesta_reciprocal_matrix(const char *fn,
                                  const double prune_threshold,
                                  int *spin_flag,
                                  int *block_size,
                                  int *nk,
                                  int *ncomp,
                                  indexType **nnz,
                                  indexType ***row_index,
                                  indexType ***col_index,
                                  cmplx ***value,
                                  char* errmsg) {

    indexType ik, icomp, no_u, io_u, jo_u,  nspin_blocks;
    indexType num_rows;
    int file_header[HEADER_SIZE];
    cmplx *buffer;
    std::size_t read_count, block_count, block_ext_count, file_count;
    FILE *file;

    /** - open the file */
    file = std::fopen(fn, "rb");
    if (!file) { 
        if (errmsg) std::sprintf(errmsg, "File %s could not be opened.\n", fn);
        return 1;
    }

    /** - read the 20 byte file headers */
    read_count = std::fread(reinterpret_cast<char*>(&file_header), sizeof(int), HEADER_SIZE, file);
    if (read_count < HEADER_SIZE || std::ferror(file)) {
        if (errmsg) std::sprintf(errmsg, "ERROR reading the file header of %s", fn);
        return 2;
    }

    /** File header information contains
     * 1. # of orbitals in the unit-cell / matrix row block size
     * 2. matrix col block size (usually identical)
     * 3. # of k-points
     * 4. # of components
     * 5. # of spin configurations 
    */
    no_u = file_header[0]; *nk = file_header[2]; *ncomp = file_header[3]; *spin_flag = file_header[4];

    /** - allocate buffer and read in the whole file */
    /** - files contains
     *    Hk(io_u, jo_u, ispin, ik)
     *    dHk/Vk(io_u, jo_u, ispin, ix, ik)
     */
    nspin_blocks = num_spin_configurations(*spin_flag);
    block_count = no_u*no_u*(*spin_flag);
    block_ext_count = no_u*no_u*nspin_blocks;
    *block_size = sqrt(block_ext_count); 



    
    file_count = block_ext_count*(*ncomp)*(*nk);
    buffer = reinterpret_cast<cmplx*>( std::malloc(file_count*sizeof(cmplx)) );
    if (*spin_flag == 3) {  /** - if there are 3 spin components in the file, add the fourth one using hermitian symmetry */
        for (ik = 0; ik < *nk; ik++) {
            for (icomp = 0; icomp < *ncomp; icomp++) {
                read_count = std::fread(reinterpret_cast<char*>(&buffer[(ik*(*ncomp) + icomp)*block_ext_count]), sizeof(cmplx),  \
                                            block_count, file);
                if (read_count < block_count || std::ferror(file)) {
                    if (errmsg) std::sprintf(errmsg, "ERROR reading the matrix elements of %s at ik = %d", fn, ik);
                    return 3;
                }
                /** - add the transposed block */
                for (jo_u = 0; jo_u < no_u; jo_u++) {
                    for (io_u = 0; io_u < no_u; io_u++) {
                        buffer[ (((ik*(*ncomp) + icomp)*nspin_blocks + 3)*no_u + jo_u)*no_u + io_u ]    \
                        = std::conj(buffer[ (((ik*(*ncomp) + icomp)*nspin_blocks + 2)*no_u + io_u)*no_u + jo_u ]);
                    }
                }
            }
        }
    }
    else {  /** - otherwise, just read the whole file immediately */
        read_count = std::fread(reinterpret_cast<char*>(&buffer[0]), sizeof(cmplx),  \
                                    file_count, file);
        if (read_count < file_count || std::ferror(file)) {
            if (errmsg) std::sprintf(errmsg, "ERROR reading the matrix elements of %s", fn);
            return 3;
        }
    }
    std::fclose(file);

    /** buffer[ (((ik*ncomp + icomp)*nspin_blocks + ispin)*no_u + jo_u)*no_u + io_u ]
     *  contains <io_u, k, sigma | O | jo_u, k, sigma'>, where
     * 
     *    ispin | sigma | sigma'
     *   ------------------------
     *      0   | up(0) | up(0)
     *      1   |down(1)|down(1)
     *      2   | up(0) |down(1)
     *      3   |down(1)| up(0)
    */

    /** - generate sparse matrix, nnz will change due to pruning */
    num_rows = no_u*(*nk) * num_spin_rows(*spin_flag);
    *nnz = reinterpret_cast<indexType*>( std::malloc((*ncomp)*sizeof(indexType)) );
    *value = reinterpret_cast<cmplx**>( std::malloc((*ncomp)*sizeof(cmplx*)) );
    *col_index = reinterpret_cast<indexType**>( std::malloc((*ncomp)*sizeof(indexType*)) );
    *row_index = reinterpret_cast<indexType**>( std::malloc((*ncomp)*sizeof(indexType*)) );
    for (icomp = 0; icomp < *ncomp; icomp++) {
        (*nnz)[icomp] = no_u*no_u*(*nk) * num_spin_configurations(*spin_flag);
        (*value)[icomp] = reinterpret_cast<cmplx*>( std::malloc((*nnz)[icomp]*sizeof(cmplx)) );
        (*col_index)[icomp] = reinterpret_cast<indexType*>( std::malloc((*nnz)[icomp]*sizeof(indexType)) );
        (*row_index)[icomp] = reinterpret_cast<indexType*>( std::malloc((num_rows+1)*sizeof(indexType)) );
    }
    create_csr_sparse_matrix(prune_threshold,
                             no_u,
                             *ncomp,
                             nspin_blocks,
                             *nk,
                             buffer,
                             *row_index,
                             *col_index,
                             *value,
                             *nnz);

    /** - re-allocate the arrays to the actual size */
    for (icomp = 0; icomp < *ncomp; icomp++) {
        (*value)[icomp] = reinterpret_cast<cmplx*>( std::realloc((*value)[icomp], (*nnz)[icomp]*sizeof(cmplx)) );
        (*col_index)[icomp] = reinterpret_cast<indexType*>( std::realloc((*col_index)[icomp], (*nnz)[icomp]*sizeof(indexType)) );
    }

    /** - free the buffer */
    std::free(buffer);

    return 0;
}


void Read_Siesta_Cropped::build_Hamiltonian(){
  std::ifstream inFile;
  std::string run_dir  = parameters().run_dir_,
    filename = parameters().filename_;

  set_sysLength(1.0);  
  set_sysSubLength(1.0);
  

  inFile.precision(14);
  inFile.open(run_dir+"operators/"+filename+".HK");

  std::cout<<run_dir+"Reading Hamiltonian on:  operators/"+filename+".HK"  <<std::endl<<std::endl;
  //"REMEMBER: vcx/hcx hack!!  Imaginary part of the Hamiltonian is being dumped on read; "<<

  std::size_t DIM;


 


  /*
  indexType outerIndexPtr[DIM+1];
  indexType innerIndices[NNZ];
  type values[NNZ];
  */

  


  

    int ierr, icomp, j, spin_flag, block_size, nk, ncomp;
    char *errmsg = new char[256];

    // "../crte2/crte2.HK"
    ierr = read_siesta_reciprocal_matrix( (run_dir+"operators/"+filename+".HK").c_str(),
                                         1.e-3,
                                         &spin_flag,
                                         &block_size,
                                         &nk,
                                         &ncomp,
                                         &HK_nnz_,
                                         &HK_row_index_,
                                         &HK_col_index_,
                                         &HK_values_,
                                         errmsg);
  

   for (icomp = 0; icomp < ncomp; icomp++) {
        std::printf("Component %d \n", icomp);
        std::printf("Number of non-zero entries: %d \n", HK_nnz_[icomp]);
        std::printf("Row indices: ");
        for (j = 0; j < 10; j++) std::printf("%d ", HK_row_index_[icomp][j]);
        std::printf("...\nColumn indices: ");
        for (j = 0; j < 20; j++) std::printf("%d ", HK_col_index_[icomp][j]);
        std::printf("...\nValues: ");
        for (j = 0; j < 20; j++) std::printf("(%.3e, %.3e) ", std::real(HK_values_[icomp][j]), std::imag(HK_values_[icomp][j]));
        std::printf("...\n");
    }



    
    DIM = nk*block_size;


  parameters().DIM_    = DIM;
  parameters().SUBDIM_ = DIM;  
  parameters().C_      = 0;
  parameters().W_      = 1;
  parameters().LE_     = DIM;
  

  
  //  Hc_.resize(DIM,DIM);
  Eigen::SparseMatrix<type, Eigen::RowMajor,indexType> Hc_tmp=Eigen::Map<Eigen::SparseMatrix<type, Eigen::RowMajor,indexType> > (DIM, DIM, HK_nnz_[0], HK_row_index_[0], HK_col_index_[0],HK_values_[0]);


  trimBlocks(Hc_, Hc_tmp, nk, block_size);

  parameters().DIM_    = DIM-nk*4;
  parameters().SUBDIM_ = DIM-nk*4;
  parameters().LE_     = DIM-nk*4;

  /*
  auto Hc_adjoint = Eigen::SparseMatrix<type, Eigen::RowMajor,indexType>(Hc_.transpose().conjugate());

  Eigen::SparseMatrix<type, Eigen::RowMajor,indexType> diff = Eigen::SparseMatrix<type, Eigen::RowMajor,indexType>((Hc_-Hc_adjoint));
  auto norm = (diff).norm();
  std::cout<<"Norm:  "<< norm<<std::endl;

  int max_k=0;
double max_norm = 0.0;
for (int k = 0; k < diff.outerSize(); ++k) {
    for (Eigen::SparseMatrix<type, Eigen::RowMajor,indexType>::InnerIterator it(Hc_, k); it; ++it) {
        double abs_val = std::abs(it.value());
        if (abs_val > max_norm){
	  max_norm = abs_val, max_k=k;
	}
    }
}

 std::cout<<"max:  "<< max_norm<<"   Max k"<< max_k<<"/"<<diff.nonZeros()<<std::endl;
  
  Hc_=(Hc_+Hc_adjoint)/2;
  */

  
  inFile.close();
};


void Read_Siesta_Cropped::trimBlocks(SpMatrixXcp& Hc, const SpMatrixXcp& Hc_tmp, int nk, int block_size) {
    int new_block_size = block_size - 4;
    int new_dim = nk * new_block_size;

    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    triplets.reserve(static_cast<size_t>(nk) * new_block_size * new_block_size);

    for (int k = 0; k < nk; ++k) {
        indexType old_block_row_start = k * block_size;
        indexType old_block_col_start = k * block_size;
        indexType new_block_row_start = k * new_block_size;
        indexType new_block_col_start = k * new_block_size;

        for (int j = 0; j < new_block_size; ++j) {
            for (SpMatrixXcp::InnerIterator it(Hc_tmp, old_block_col_start + j); it; ++it) {
                indexType i = it.row();
                indexType j_global = it.col();

                if (i < old_block_row_start + new_block_size) {  // only row check needed
                    indexType new_i = new_block_row_start + (i - old_block_row_start);
                    indexType new_j = new_block_col_start + j; // j is already < new_block_size
                    triplets.emplace_back(new_i, new_j, it.value());
                }
            }
        }
    }

    Hc.resize(new_dim, new_dim);
    Hc.setFromTriplets(triplets.begin(), triplets.end());
}



void Read_Siesta_Cropped::vel_op_x (type vec[], type p_vec[]){
  int Dim = this->parameters().DIM_;
  
  Eigen::Map<VectorXdT> eig_vec(vec,Dim),
    eig_p_vec(p_vec, Dim);

 
  eig_vec = vxc_ * eig_p_vec;

 
};


void Read_Siesta_Cropped::vel_op_y (type vec[], type p_vec[]){
  int Dim = this->parameters().DIM_;
  
  Eigen::Map<VectorXdT> eig_vec(vec,Dim),
    eig_p_vec(p_vec, Dim);

 
  eig_vec = vyc_ * eig_p_vec;
};


void Read_Siesta_Cropped::vel_op_szvy (type vec[], type p_vec[]){
  int Dim = this->parameters().DIM_;
  
  Eigen::Map<VectorXdT> eig_vec(vec,Dim),
    eig_p_vec(p_vec, Dim);

 
  eig_vec = szvy_ * eig_p_vec;
};



void Read_Siesta_Cropped::H_ket ( type* vec, type* p_vec ){
  H_ket(vec, p_vec, damp_op(), dis());
};



void Read_Siesta_Cropped::H_ket ( type* vec, type* p_vec, r_type* dmp_op, r_type* dis_vec) {
  int Dim = this->parameters().DIM_,
      subDim = this->parameters().SUBDIM_,
      W = this->parameters().W_,
      C = this->parameters().C_;
  
  /*
#pragma omp parallel for
  for(int i = 0; i < Dim; i++)
    p_vec[ i ] *= dmp_op[ i ];
  */

  Eigen::Map<VectorXdT> eig_vec(vec,Dim),
    eig_p_vec(p_vec, Dim);
  
  if(H_.size()>0)
    eig_vec = H_ * eig_p_vec;
  
  else if(Hc_.size()>0)  
    eig_vec = Hc_ * eig_p_vec;

  /*
#pragma omp parallel for
  for(int i = 0; i < subDim; i++) 
    vec[ i + C * W ]    +=  dis_vec[ i ] * p_vec[ i + C * W ]/a_;
  */
}

void Read_Siesta_Cropped::update_cheb ( type vec[], type p_vec[], type pp_vec[]){
  update_cheb ( vec, p_vec, pp_vec, damp_op(), NULL);
};

  
void Read_Siesta_Cropped::update_cheb ( type vec[], type p_vec[], type pp_vec[], r_type damp_op[], r_type*){

 
  int Dim = this->parameters().DIM_;
  /*
#pragma omp parallel for
  for(int i = 0; i < Dim; i++)
    pp_vec[ i ] *= damp_op[ i ] * damp_op[ i ];
  */

  
  Eigen::Map<VectorXdT> eig_vec(vec,Dim),
    eig_p_vec(p_vec, Dim),
    eig_pp_vec(pp_vec, Dim);


    
 


  if(H_.size()>0)
    eig_vec = 2.0 * H_ * eig_p_vec - eig_pp_vec;
  
  if(Hc_.size()>0)
    eig_vec = 2.0 * Hc_ * eig_p_vec - eig_pp_vec;
  
    
  eig_pp_vec = eig_p_vec;
  eig_p_vec = eig_vec;

   

      
}



void Read_Siesta_Cropped::damp ( r_type damp_op[]){

  set_damp_op(damp_op);
  
  int Dim = this->parameters().DIM_;
 
  SpMatrixXp Id(Dim,Dim), gamma(Dim,Dim);//, dis(Dim,Dim);  dis.setZero();
  Id.setIdentity();
  gamma = Id;

  /*
  #pragma omp parallel for
  for(int i=0; i<Dim;i++)
    gamma.coeffRef(i,i) *=damp_op[ i ];
  
  if(H_.size()>0)
    H_ = gamma*H_;
  if(Hc_.size()>0)
    Hc_ = gamma*Hc_;
  */
}




void Read_Siesta_Cropped::setup_velOp(){
  std::ifstream inFile;
  std::string run_dir  = parameters().run_dir_,
    filename = parameters().filename_;

  inFile.open(run_dir+"operators/"+filename+".VK");

  //std::cout<<"  /Remember vx/vcx hack. /real part of the Velocity is being dumped on read;"<<std::endl<<std::endl;
    
    
  std::size_t DIM, NNZ;
  


  std::cout<<run_dir+"Reading Velocity operator on:  operators/"+filename+".VK"  <<std::endl<<std::endl;

    int ierr, icomp, j, spin_flag, block_size, nk, ncomp;
    //indexType *nnz, **row_index, **col_index;
    //std::complex<double> **values;
    char *errmsg = new char[256];

    // "../crte2/crte2.HK"
    ierr = read_siesta_reciprocal_matrix( (run_dir+"operators/"+filename+".VK").c_str(),
                                         1.e-3,
                                         &spin_flag,
                                         &block_size,
                                         &nk,
                                         &ncomp,
                                         &VK_nnz_,
                                         &VK_row_index_,
                                         &VK_col_index_,
                                         &VK_values_,
                                         errmsg);

    if(ierr!=0){
      std::cout<<errmsg<<std::endl;
      return;
    }
    
    DIM = nk*block_size;


   for (icomp = 0; icomp < ncomp; icomp++) {
        std::printf("VECLOCITY Component %d \n", icomp);
        std::printf("Number of non-zero entries: %d \n", VK_nnz_[icomp]);
        std::printf("Row indices: ");
        for (j = 0; j < 10; j++) std::printf("%d ", VK_row_index_[icomp][j]);
        std::printf("...\nColumn indices: ");
        for (j = 0; j < 20; j++) std::printf("%d ", VK_col_index_[icomp][j]);
        std::printf("...\nValues: ");
        for (j = 0; j < 20; j++) std::printf("(%.3e, %.3e) ", std::real(VK_values_[icomp][j]), std::imag(VK_values_[icomp][j]));
        std::printf("...\n");
   }

   /*   
      for (icomp = 0; icomp < ncomp; icomp++) {
        std::printf("Searching for NaNs: Component %d \n", icomp);
        std::printf("Number of non-zero entries: %d \n", VK_nnz_[icomp]);


	for (int j = 0; j < VK_nnz_[icomp]; j++){
	  std::complex<double> val = VK_values_[icomp][j];
          if (abs(val)>270 ){
	    std::printf("(%d, %.3e, %.3e) ",j, std::real(VK_values_[icomp][j]), std::imag(VK_values_[icomp][j]));
	    std::printf("...\n");
	  }
	}
   	
    }
   */
    
  Eigen::SparseMatrix<type, Eigen::RowMajor,indexType> vxc_tmp=Eigen::Map<Eigen::SparseMatrix<type, Eigen::RowMajor,indexType> > (DIM, DIM, VK_nnz_[0], VK_row_index_[0], VK_col_index_[0],VK_values_[0]);
  Eigen::SparseMatrix<type, Eigen::RowMajor,indexType> vyc_tmp=Eigen::Map<Eigen::SparseMatrix<type, Eigen::RowMajor,indexType> > (DIM, DIM, VK_nnz_[1], VK_row_index_[1], VK_col_index_[1],VK_values_[1]);
  

  
  trimBlocks(vxc_, vxc_tmp, nk, block_size);
  trimBlocks(vyc_, vyc_tmp, nk, block_size);



  this->setup_spinCurrentOp();
  //vxc_.setIdentity();


};



void Read_Siesta_Cropped::setup_spinCurrentOp(){
  std::ifstream inFile;
  std::string run_dir  = parameters().run_dir_,
    filename = parameters().filename_;

  inFile.open(run_dir+"operators/"+filename+".VK");

  //std::cout<<"  /Remember vx/vcx hack. /real part of the Velocity is being dumped on read;"<<std::endl<<std::endl;
    
    
  std::size_t DIM, NNZ;
  


  std::cout<<run_dir+"Reading Spin Current operator on:  operators/"+filename+".JSK"  <<std::endl<<std::endl;

    int ierr, icomp, j, spin_flag, block_size, nk, ncomp;
    //indexType *nnz, **row_index, **col_index;
    //std::complex<double> **values;
    char *errmsg = new char[256];

    // "../crte2/crte2.HK"
    ierr = read_siesta_reciprocal_matrix( (run_dir+"operators/"+filename+".JSK").c_str(),
                                         1.e-3,
                                         &spin_flag,
                                         &block_size,
                                         &nk,
                                         &ncomp,
                                         &JSK_nnz_,
                                         &JSK_row_index_,
                                         &JSK_col_index_,
                                         &JSK_values_,
                                         errmsg);

    if(ierr!=0){
      std::cout<<errmsg<<std::endl;
      return;
    }
    
    DIM = nk*block_size;


    

   for (icomp = 0; icomp < ncomp; icomp++) {
        std::printf("SPIN CURRENT Component %d \n", icomp);
        std::printf("Number of non-zero entries: %d \n", VK_nnz_[icomp]);
        std::printf("Row indices: ");
        for (j = 0; j < 10; j++) std::printf("%d ", VK_row_index_[icomp][j]);
        std::printf("...\nColumn indices: ");
        for (j = 0; j < 20; j++) std::printf("%d ", VK_col_index_[icomp][j]);
        std::printf("...\nValues: ");
        for (j = 0; j < 20; j++) std::printf("(%.3e, %.3e) ", std::real(VK_values_[icomp][j]), std::imag(VK_values_[icomp][j]));
        std::printf("...\n");
   }

    
  Eigen::SparseMatrix<type, Eigen::RowMajor,indexType> szvy_tmp=Eigen::Map<Eigen::SparseMatrix<type, Eigen::RowMajor,indexType> > (DIM, DIM, JSK_nnz_[2], JSK_row_index_[2], JSK_col_index_[2],JSK_values_[2]);
    

  

  trimBlocks(szvy_, szvy_tmp, nk, block_size);



  //vxc_.setIdentity();


};






void Read_Siesta_Cropped::update_dis ( r_type dis_vec[], r_type damp_op[]){
  int subDim = this->parameters().SUBDIM_;
  int C   = this->parameters().C_,
      W   = this->parameters().W_;

  set_dis(dis_vec);

  if(H_.size()>0){
  #pragma omp parallel for
  for(int i=0; i<subDim;i++)
     H_.coeffRef(C*W + i, C*W +i) = damp_op[i] * b_/a_;
     
  
  #pragma omp parallel for
  for(int i=0; i<subDim;i++)
     H_.coeffRef(C*W + i, C*W +i) += damp_op[i] * dis_vec[i]/a_;
  }

  /*
  if(Hc_.size()>0){
  #pragma omp parallel for
  for(int i=0; i<subDim;i++)
     Hc_.coeffRef(C*W + i, C*W +i) = damp_op[i] * b_/a_;
     
  
  #pragma omp parallel for
  for(int i=0; i<subDim;i++)
     Hc_.coeffRef(C*W + i, C*W +i) += damp_op[i] * dis_vec[i]/a_;
  }
  */
}


