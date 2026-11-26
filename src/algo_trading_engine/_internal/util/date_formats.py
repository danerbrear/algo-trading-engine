class DateFormats:
    """Common strftime/strptime patterns for datetimes.

    Multiple attributes may share the same pattern; pick the name that matches
    the use (expiration vs cache path vs generic ISO date).
    """

    EXPIRATION = "%Y-%m-%d"
    ISO_DATE = "%Y-%m-%d"
    CACHE_DATE = "%Y-%m-%d"
    PRINTABLE = "%Y-%m-%d %H:%M"
    CACHE_TIME = "%H%M"
    COMPACT_DATE = "%Y%m%d"
