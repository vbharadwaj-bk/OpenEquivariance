if [ -z "$PYTHON" ]; then
    PYTHON=python3
fi

rm -rf fast_tp/extlib; mkdir fast_tp/extlib;
rm -rf build; mkdir build; pushd build; 
cmake ../fast_tp/extension -Dpybind11_DIR=$($PYTHON -m pybind11 --cmakedir)
make -j4
cmake --install . --prefix ../fast_tp/extlib
popd;
rm -rf build;

if [ -z "$NOINSTALL" ]; then
    $PYTHON -m pip install . -vv --no-deps
fi