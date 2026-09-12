"""Typed failures emitted by external evidence boundaries."""


class SourceError(RuntimeError):
    """Base class for controlled source failures."""


class SourcePolicyRejected(SourceError):
    """The request violates source-access policy."""


class SourceTimeout(SourceError):
    """The source did not respond before its deadline."""


class SourceRateLimited(SourceError):
    """The source rejected the request because of rate limits."""


class SourceUnavailable(SourceError):
    """The source returned a transport or service failure."""


class SourceResponseTooLarge(SourceError):
    """The source exceeded the configured response budget."""


class SourceFormatInvalid(SourceError):
    """The source returned an unexpected or malformed representation."""