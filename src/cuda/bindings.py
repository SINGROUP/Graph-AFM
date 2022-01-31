
import os
import torch
from torch.utils.cpp_extension import load

compile_ok = False
if torch.cuda.is_available():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    peak_find = load(name='peak_find', sources=[
        os.path.join(this_dir, 'peak_find.cpp'),
        os.path.join(this_dir, 'matching_cuda.cu'),
        os.path.join(this_dir, 'ccl_cuda.cu')],
        extra_include_paths=[this_dir]
    )
    tm_methods = peak_find.TMMethod.__members__
    compile_ok = True

def _check_compile():
    if not compile_ok:
        raise RuntimeError('Cuda extensions were not compiled correctly. Is cuda available?')

def match_template(array, template, method='mad'):
    _check_compile()
    if not (isinstance(array, torch.Tensor) and isinstance(template, torch.Tensor)):
        raise ValueError('Input arrays must be torch tensors.')
    try:
        method = tm_methods[method]
    except KeyError:
        raise ValueError(f'Unknown matching method `{method}`.')
    return peak_find.match_template(array, template, method)

def ccl(array):
    _check_compile()
    return peak_find.ccl(array.int())

def find_label_min(array, labels):
    _check_compile()
    assert array.shape == labels.shape
    return peak_find.find_label_min(array, labels)
