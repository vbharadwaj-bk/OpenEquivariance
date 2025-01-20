rm -rf extlib; mkdir extlib;
rm -rf build; mkdir build; pushd build; 
cmake .. -Dpybind11_DIR=$($PYTHON -m pybind11 --cmakedir)
make -j4
cmake --install . --prefix ../extlib
popd;
rm -rf build;

#$PYTHON -m pip install . -vv