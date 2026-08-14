
SHELL := /bin/zsh
DIR := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))


###############
#### BUILD ####
###############
clean:
	@rm -rf ./build


quick: FORCE headers
	@python setup.py build_ext --inplace


build: clean Persistence Sampling
	@python setup.py build_ext --inplace > build.log 2>&1


CXX=clang++
INSTALL_DIR=/usr/local
INSTALL_LFLAGS=-L$(INSTALL_DIR)/lib -Wl,-rpath,$(INSTALL_DIR)/lib

headers:
	@sudo rm -rf $(INSTALL_DIR)/include/ATEAMS
	@sudo mkdir $(INSTALL_DIR)/include/ATEAMS
	@sudo cp -r ateams/common.h $(INSTALL_DIR)/include/ATEAMS/
	@sudo cp -r ateams/arithmetic/include/PHAT $(INSTALL_DIR)/include/phat
	@sudo cp -r ateams/arithmetic/include/SparseRREF/ $(INSTALL_DIR)/include/SparseRREF/
	@sudo cp -r ateams/arithmetic/Persistence.h $(INSTALL_DIR)/include/ATEAMS/
	@sudo cp -r ateams/arithmetic/Sampling.h $(INSTALL_DIR)/include/ATEAMS/
	@sudo cp -r ateams/arithmetic/util.h $(INSTALL_DIR)/include/ATEAMS/


Persistence_LFLAGS = -I$(INSTALL_DIR)/include/ -L$(INSTALL_DIR)/lib -lspasm `pkg-config --libs --cflags flint mimalloc` -shared -fPIC -fexperimental-library
Persistence_CFLAGS = -O3 -std=c++20

Persistence: headers
	@sudo $(CXX) $(Persistence_LFLAGS) $(Persistence_CFLAGS) -o $(INSTALL_DIR)/lib/libATEAMS_Persistence.so ateams/arithmetic/Persistence.cpp $(INSTALL_LFLAGS)



Sampling_LFLAGS = -I$(INSTALL_DIR)/include/ -L$(INSTALL_DIR)/lib `pkg-config --libs --cflags flint mimalloc` -shared -fPIC -fexperimental-library
Sampling_CFLAGS = -O3 -std=c++20

Sampling: headers
	@sudo $(CXX) $(Sampling_LFLAGS) $(Sampling_CFLAGS) -o $(INSTALL_DIR)/lib/libATEAMS_Sampling.so ateams/arithmetic/Sampling.cpp $(INSTALL_LFLAGS)



#################
#### PROFILE ####
#################
Glauber: FORCE
	@cd test && ./profile.models.Glauber.sh 9 12 4
	@cd test && ./profile.models.Glauber.sh 99 102 2

SwendsenWang: FORCE
	@cd test && ./profile.models.SW.sh 4 7 4
	@cd test && ./profile.models.SW.sh 49 52 2

Nienhuis: FORCE
	@cd test && ./profile.models.NH.sh 49 52 2
	@cd test && ./profile.models.NH.sh 9 12 3

InvadedCluster: FORCE
	@cd test && ./profile.models.IC.sh 4 5 4
	@cd test && ./profile.models.IC.sh 19 22 2

Bernoulli: FORCE
	@cd test && ./profile.models.Bernoulli.sh 4 7 4
	@cd test && ./profile.models.Bernoulli.sh 19 22 2

BernoulliSite: FORCE
	@cd test && ./profile.models.BernoulliSite.sh 3 6 4
	@cd test && ./profile.models.BernoulliSite.sh 6 9 2

InvasionPercolation: FORCE
	@cd test && ./profile.models.IP.sh 6 9 4
	@cd test && ./profile.models.IP.sh 99 103 2


profile: Glauber SwendsenWang Nienhuis InvadedCluster BernoulliSite

test: FORCE
	@cd test && ./test.models.BernoulliSite.sh

gauntlet: FORCE test profile

# Kills all the profiles (in case something goes wrong, or is taking too long).
killall: FORCE
	@ps aux | grep -e python -e make -e profile.models | awk '{print $$2}' | xargs kill




#######################
#### DOCUMENTATION ####
#######################
tables: FORCE
	@echo "This will hang if you aren't logged in as anthony on Pangolin."
	@cd test/stats && ./stats.sync.sh && ./stats.tables.sh

docs: FORCE quick refs
	@pdoc ateams --force --html --template-dir docs/templates --output-dir=docs
	@rsync -a docs/ateams/ docs/
	@rm -rf docs/ateams
	@open "file://$(DIR)docs/index.html"

refs: FORCE
	@pandoc -t markdown_strict --citeproc _refs.md -o refs.md
	@cat _README.md refs.md > README.md

contribute: build profile docs refs





#################
#### INSTALL ####
#################
install: dependencies _install profile

dependencies: FORCE headers
	@pip install -r requirements

_install: FORCE build
	pip install -e --no-build-isolation .

FORCE:
