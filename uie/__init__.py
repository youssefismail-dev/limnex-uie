"""Limnex URL Intelligence Engine (Limnex UIE).

    import uie
    engine = uie.load("limnex-uie-e16_b5.pt", known_urls="limnex-uie-known-urls.bloom.gz")
    engine.generate("phishing", n=10)
    engine.score(["example.com/login"])

Synthetic URLs for defensive research only -- read the README's Responsible Use and No Liability
sections first. The public names are loaded on first use, so importing a submodule (for example
the corpus tools) does not import the model stack.
"""
__version__ = "0.1.0"
__all__ = ["Engine", "NOTICE", "load", "load_blocklist", "__version__"]


def __getattr__(name):
    if name in ("Engine", "NOTICE", "load", "load_blocklist"):
        from uie import api
        return getattr(api, name)
    raise AttributeError(f"module 'uie' has no attribute {name!r}")
