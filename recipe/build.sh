if [ -z "$PYTHON" ]; then
    PYTHON=python3
fi

rm -rf openequivariance/extlib; mkdir openequivariance/extlib;
rm -rf build; mkdir build; pushd build; 
cmake ../openequivariance/extension -Dpybind11_DIR=$($PYTHON -m pybind11 --cmakedir)
make -j4
cmake --install . --prefix ../openequivariance/extlib
popd;
rm -rf build;

if [ -z "$NOINSTALL" ]; then
    $PYTHON -m pip install . -vv --no-deps
fi