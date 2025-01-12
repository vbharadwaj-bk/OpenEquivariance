import numpy as np
from jinja2 import Environment, PackageLoader, FileSystemLoader 

from src.implementations.TensorProduct import TensorProduct
from src.implementations.e3nn_lite import Irreps

def raise_helper(msg):
    raise Exception(msg)

def divide(numerator, denominator):
    return numerator // denominator 

def sizeof(dtype):
    if dtype in ["float", "int", "unsigned int"]:
        return 4
    else:
        raise Exception("Provided undefined datatype to sizeof!")

def get_jinja_environment():
    env = Environment(loader=FileSystemLoader("src/templates"), extensions=['jinja2.ext.do'])
    env.globals['raise'] = raise_helper 
    env.globals['divide'] = divide 
    env.globals['sizeof'] = sizeof 
    env.globals['range'] = range
    env.globals['enumerate'] = enumerate 
    env.globals['len'] = len
    return env