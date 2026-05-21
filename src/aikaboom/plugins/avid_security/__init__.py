"""AVID/security risk analysis plugin."""
from aikaboom.plugins import register
from aikaboom.plugins.avid_security.plugin import AvidSecurityPlugin

# Self-register on import.
register(AvidSecurityPlugin())
