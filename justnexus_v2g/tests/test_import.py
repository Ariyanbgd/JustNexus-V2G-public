def test_import():
    import justnexus_v2g

def test_import_tariffs_module():
    from justnexus_v2g.exogenous.tariffs import (
        export_tariff_profile,
        import_tariff_profile,
    )

    assert callable(import_tariff_profile)
    assert callable(export_tariff_profile)
