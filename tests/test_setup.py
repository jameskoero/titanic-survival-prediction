import importlib


def test_required_ml_dependencies_are_importable():
    for module_name in ('pandas', 'numpy', 'sklearn', 'matplotlib'):
        assert importlib.import_module(module_name) is not None
