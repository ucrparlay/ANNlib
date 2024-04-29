CXX = g++
#CXX = clang++
DIR_FMWK = ../
DIR_PARLAY = parlaylib/include
DIR_CPAM = CPAM/include
DIR_CPAM_ASPEN = CPAM/examples/graphs/aspen
DIR_INCLUDE = $(DIR_FMWK) $(DIR_PARLAY) $(DIR_CPAM) $(DIR_CPAM_ASPEN)

CXXFLAGS_BASIC = -std=c++17 -Wall -Wextra -march=native -pthread $(DIR_INCLUDE:%=-I%)
ifeq (DEBUG, $(MODE))
	CXXFLAGS_BASIC += -O0 -g -DDEBUG_OUTPUT
else
	CXXFLAGS_BASIC += -O3
endif
CXXFLAGS_JEMALLOC =
ifneq (, $(shell which jemalloc-config))
	LIBDIR_JEMALLOC := $(shell jemalloc-config --libdir)
	CXXFLAGS_JEMALLOC := -L$(LIBDIR_JEMALLOC) -ljemalloc -Wl,-rpath=$(LIBDIR_JEMALLOC)
endif

CXXFLAGS_HDF5 =
ifeq (ON, $(HDF5))
	CXXFLAGS_HDF5 = -DSUPPORT_HDF5 -lhdf5 -lhdf5_cpp
endif

calc_recall: calc_recall.cpp
	$(CXX) $< $(CXXFLAGS_BASIC) $(CXXFLAGS_RELEASE) $(CXXFLAGS_HDF5) $(CXXFLAGS_JEMALLOC) -o $@
