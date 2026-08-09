def test_generator_initialization():
    from lokum_engine.rag.generator import LocalGenerator
    # Just testing instantiation to avoid downloading huge models in CI
    gen = LocalGenerator(model_id="mock_model", lazy_load=True)
    assert gen.model_id == "mock_model"
    assert gen.model is None
