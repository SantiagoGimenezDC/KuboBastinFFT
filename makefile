CXX := g++


NAME   = KuboBastinFFT
BINARY = KuboBastinFFT.bin




WARNING := -Wall -Wextra

#-------------Debug build flags----------#
Debug_LDFLAGS := -L/usr/local/lib -lfftw3  -std=c++17 -g  
Debug_CFLAGS := -I/usr/local/include -Isrc/Includes -g -std=c++17 $(WARNING)
#-----------------------------------------------#


#-------------Standard build flags----------#
STD_LDFLAGS := -L/usr/local/lib -L/lib -lfftw3 -lmemkind
STD_CFLAGS  := -I/usr/local/include  -O3 -msse2  -std=c++17 $(WARNING)
#-----------------------------------------------#





BUILDDIR  := build
SOURCEDIR := src
HEADERDIR := src




cppSOURCES := $(shell find $(SOURCEDIR) -name '*.cpp')
cppOBJECTS := $(addprefix $(BUILDDIR)/, $(cppSOURCES:%.cpp=%.o))
cppDEPENDS := $(addprefix $(BUILDDIR)/, $(cppSOURCES:%.cpp=%.d))

DEPENDS    := $(addprefix $(BUILDDIR)/, $(cppSOURCES:%.cpp=%.d))





.PHONY : standard debug clean

debug : CXXLAGS = $(Debug_CFLAGS)
debug : LDFLAGS = $(Debug_LDFLAGS)
debug : $(NAME)


standard : CXXFLAGS = $(STD_CFLAGS)
standard : LDFLAGS  = $(STD_LDFLAGS)
standard : $(NAME)





$(NAME) : $(cppOBJECTS) 
	$(CXX) $(cppOBJECTS) $(LDFLAGS) -o $@ -fopenmp 


-include $(cppDEPENDS)

$(BUILDDIR)/%.o : %.cpp makefile
	mkdir -p $(dir $@)
	$(CXX) -I$(HEADERDIR) -I$(dir $<) $(CXXFLAGS) -MMD -MP  -c $< -o $@ -fopenmp -lm



clean:
	$(RM) $(NAME) $(cppOBJECTS) $(DEPENDS) 


#DEPENDS, -include, -MMD -MP options  taken from:
#https://stackoverflow.com/questions/52034997/how-to-make-makefile-recompile-when-a-header-file-is-changed

#build dir mirroring rule ( basically, just "mkdir -p $(dir $@)"  ) taken from:
#https://stackoverflow.com/questions/5650765/makefile-mirror-build-directory
