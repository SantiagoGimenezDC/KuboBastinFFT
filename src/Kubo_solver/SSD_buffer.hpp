#ifndef SSD_BUFFER_HPP
#define SSD_BUFFER_HPP

#include<iostream>
#include<fstream>
#include<string>

#include<string>
#include"../static_vars.hpp"
#include<eigen-3.4.0/Eigen/Core>
#include<fcntl.h>
#include<unistd.h>
#include <cstdio>


#include <fixed_allocator.h>
#include <sys/mman.h>
#include <memkind.h>

static void print_err_message(int err)
{
    char error_message[MEMKIND_ERROR_MESSAGE_SIZE];
    memkind_error_message(err, error_message, MEMKIND_ERROR_MESSAGE_SIZE);
    fprintf(stderr, "%s\n", error_message);
}

class SSD_buffer{
private:
  int COLS_, ROWS_, ROWS_stride_, COLS_stride_, ROWS_rest_, COLS_rest_, num_write_buffers_, num_read_buffers_;
  std::size_t  SSD_size_, RAM_size_;
  std::string filename_;
  bool rest_buffer_=false;

  struct memkind *pmem_kind_unlimited_ = NULL;  
  void* addr_;
  type *SSD_buffer_;

  
  FILE* file_;
  FILE* out_, *in_;
  
public:
  ~SSD_buffer(){
    memkind_free(pmem_kind_unlimited_, SSD_buffer_);

    int err = memkind_destroy_kind(pmem_kind_unlimited_);
    if (err) {
        print_err_message(err);
        return;
    }
    
    munmap(addr_, std::size_t(SSD_size_));
  };
  
  int COLS_stride(){return COLS_stride_;};
  int ROWS_stride(){return ROWS_stride_;};
  int num_buffers(){return num_read_buffers_;};

  int COLS_rest(){return COLS_rest_;};
  
  SSD_buffer(int COLS, int ROWS, std::size_t RAM_size, std::string filename ) : COLS_(COLS), ROWS_(ROWS), RAM_size_(RAM_size), filename_(filename){

    SSD_size_ =  double(COLS_) * double(ROWS_) * sizeof(type);
    
    std::size_t SSD_size_int = static_cast<size_t>(SSD_size_),
                RAM_size_int = static_cast<size_t>(RAM_size_);


    
    ROWS_stride_ = RAM_size_int /( sizeof(type) * COLS_) ;
    COLS_stride_ = RAM_size_int /( sizeof(type) * ROWS_);
      
    num_write_buffers_ = static_cast<int>( COLS_ / COLS_stride_  );
    num_read_buffers_  = static_cast<int>( ROWS_ / ROWS_stride_  );

    ROWS_rest_   = ROWS_  -  num_read_buffers_  * ROWS_stride_ ;
    COLS_rest_   = COLS_  -  num_write_buffers_ * COLS_stride_;


    std::cout<<COLS_stride_<<" "<<COLS_rest_<<"      "<<ROWS_stride_<<" "<<ROWS_rest_<<"    "<< SSD_size_int <<std::endl;



    std::string path = "/mnt/mem/";
    int status = memkind_check_dax_path(path.c_str());
    if (!status)
        fprintf(stdout, "PMEM kind %s is on DAX-enabled file system.\n", path);
    else 
        fprintf(stdout, "PMEM kind %s is not on DAX-enabled file system.\n", path);
    


    
    //addr_ = mmap(NULL, SSD_size_int, PROT_READ | PROT_WRITE,
    //                MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

    
    int err = memkind_create_pmem(path.c_str(), 0, &pmem_kind_unlimited_);
    //int err = memkind_create_fixed(addr_, SSD_size_int * std::size_t(sizeof(type)), &pmem_kind_unlimited_);

    if (err) {
        print_err_message(err);
        return;
    }
    
    SSD_buffer_ = (type *) memkind_malloc(pmem_kind_unlimited_, SSD_size_int * std::size_t(sizeof(type)) );
    
    std::cout<<"RAM buffer size:  "<<RAM_size_/1E9 <<"GBs   SSD buffer size: "<< SSD_size_/1E9<<"GBs"<<std::endl;
    std::cout<<"Number of SSD writes:  "<<num_write_buffers_ <<";  Number of SSD reads: "<< num_read_buffers_<<std::endl<<std::endl;

  };


  void reset_buffer(){};


  
  void upload_col_buffer_to_SSD(int buffer_num, type* RAM_buffer){
    
    std::size_t buffer_size = ( buffer_num == num_write_buffers_ ? COLS_rest_ : COLS_stride_);

    if(buffer_size == 0)
      return;
    
    buffer_size *= ROWS_;

    if(buffer_num > num_write_buffers_){
      std::cout<<"invalid buffer num:  "<<buffer_num<<"/"<<num_write_buffers_<<std::endl;
      return;
    }
  
    
    for(std::size_t i = 0; i < buffer_size ; i ++ )
      SSD_buffer_[buffer_num * COLS_stride_ * ROWS_ + i ] = RAM_buffer[i];
    
  }
  

  
  
  int retrieve_row_buffer_from_SSD(int buffer_num, type RAM_buffer[] ){


    std::size_t buffer_size = ( buffer_num == num_read_buffers_ ? ROWS_rest_ : ROWS_stride_);

    if(buffer_size == 0)
      return 0;

      
    for(int j = 0; j < COLS_; j++)
      for(std::size_t i = 0; i < buffer_size; i++)
	RAM_buffer[buffer_size * j  + i] = SSD_buffer_[ j * ROWS_ + buffer_num * ROWS_stride_ + i ];
      

    
 
    return buffer_size ;
  }
  
};


#endif //SSD_BUFFER_HPP
