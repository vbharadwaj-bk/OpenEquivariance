if [ -z "$PYTHON" ]; then
    PYTHON=python3
fi

rm -rf extlib; mkdir extlib;
rm -rf build; mkdir build; pushd build; 
cmake .. -Dpybind11_DIR=$($PYTHON -m pybind11 --cmakedir)
make -j4
cmake --install . --prefix ../extlib
popd;
rm -rf build;

if [ -z "$NOINSTALL" ]; then
    $PYTHON -m pip install . -vv
fi