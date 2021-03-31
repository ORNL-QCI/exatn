#sudo apt install libboost-all-dev
#add --user to pip3 install if needed
pip3 install --upgrade pip
pip3 install setuptools wheel
pip3 install networkx opt_einsum optuna ray autoray
pip3 install kahypar==1.1.6
git clone --recursive https://github.com/jcmgray/cotengra.git ./tpls/cotengra
cd ./tpls/cotengra
git submodule init
git submodule update --init --recursive
pip3 install .
cd ../..
