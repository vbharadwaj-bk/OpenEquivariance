rm -rf build; mkdir build; pushd build; 
cmake .. -Dpybind11_DIR=$($PYTHON -m pybind11 --cmakedir)
make -j4
cmake --install . --prefix ..
rm -rf build;
popd;

$PYTHON -m pip install . -vv