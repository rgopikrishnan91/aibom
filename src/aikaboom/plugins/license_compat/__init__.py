"""License-compatibility analysis plugin."""
from aikaboom.plugins import register
from aikaboom.plugins.license_compat.plugin import LicenseCompatPlugin

# Self-register on import.
register(LicenseCompatPlugin())
