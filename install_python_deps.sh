#sudo apt install libboost-all-dev
pip3 install setuptools wheel
pip3 install networkx kahypar opt_einsum optuna ray autoray
git clone --recursive https://github.com/jcmgray/cotengra.git ./tpls/cotengra
cd ./tpls/cotengra
git submodule init
git submodule update --init --recursive
pip3 install .
cd ../..
