"""Recursive walks expose --min-trust and --regen-on-low-trust controls."""
import inspect


def test_recursive_bom_accepts_min_trust_kwarg():
    from aikaboom.utils.recursive_bom import generate_recursive_boms
    sig = inspect.signature(generate_recursive_boms)
    assert "min_trust" in sig.parameters
    assert "regen_on_low_trust" in sig.parameters
    assert "cache_policy" in sig.parameters


def test_recursive_bom_min_trust_defaults_to_zero():
    from aikaboom.utils.recursive_bom import generate_recursive_boms
    sig = inspect.signature(generate_recursive_boms)
    assert sig.parameters["min_trust"].default == 0.0
    assert sig.parameters["regen_on_low_trust"].default is False
    assert sig.parameters["cache_policy"].default == "use"
