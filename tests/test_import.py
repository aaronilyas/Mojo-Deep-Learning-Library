import mojodl


def test_import_loads_compiled_engine():
    assert mojodl._engine.__file__.endswith(".so")
