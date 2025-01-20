if [ -z "$PYTHON" ]; then
    PYTHON=python3
fi

rm -rf src/extlib; mkdir src/extlib;
rm -rf build; mkdir build; pushd build; 
cmake ../src/extension -Dpybind11_DIR=$($PYTHON -m pybind11 --cmakedir)
make -j4
cmake --install . --prefix ../src/extlib
popd;
rm -rf build;

if [ -z "$NOINSTALL" ]; then
    $PYTHON -m pip install . -vv
fi