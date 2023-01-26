#CC := clang++
CXX := g++

#-------------Debug build flags----------#
Debug_LDFLAGS := -L/usr/local/lib -lfftw3 -lm 
Debug_CFLAGS := -I/usr/local/include -Isrc/Includes -Wall -g -std=c++17
#-----------------------------------------------#


#-------------Standard build flags----------#
STD_LDFLAGS := -L/usr/local/lib -lfftw3 -lm -std=c++17
STD_CFLAGS  := -I/usr/local/include  -O3 -msse2  -std=c++17
#-----------------------------------------------#

WARNING := -Wall -Wextra

BUILDDIR := build
SOURCEDIR := src
HEADERDIR := src



cppSOURCES := $(shell find $(SOURCEDIR) -name '*.cpp')
cppOBJECTS := $(addprefix $(BUILDDIR)/, $(cppSOURCES:%.cpp=%.o))
cppDEPENDS := $(addprefix $(BUILDDIR)/, $(cppSOURCES:%.cpp=%.d))


DEPENDS := $(addprefix $(BUILDDIR)/, $(cppSOURCES:%.cpp=%.d))

NAME = quasar
BINARY = quasar.bin


.PHONY : standard debug clean

debug : CXXLAGS=$(Debug_CFLAGS)
debug : LDFLAGS=$(Debug_LDFLAGS)
debug : quasar


standard : CXXFLAGS=$(STD_CFLAGS)
standard : LDFLAGS=$(STD_LDFLAGS)
standard : quasar



quasar : $(cppOBJECTS) 
	$(CXX) $(cppOBJECTS)  $(WARNING) $(LDFLAGS) -o $@ -fopenmp

-include $(cppDEPENDS)

$(BUILDDIR)/%.o : %.cpp makefile 
	$(CXX) -I$(HEADERDIR) -I$(dir $<) $(WARNING) $(CXXFLAGS) -MMD -MP  -c $< -o $@ -fopenmp



clean:
	$(RM) quasar $(cppOBJECTS) $(DEPENDS) 


#DEPENDS, -include, -MMD -MP options  taken from:
#https://stackoverflow.com/questions/52034997/how-to-make-makefile-recompile-when-a-header-file-is-changed
