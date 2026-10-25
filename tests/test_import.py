import mojodl


def test_import_loads_compiled_engine():
    assert mojodl._engine.__file__.endswith(".so")


def test_answer_from_mojo():
    assert mojodl.answer() == 42
