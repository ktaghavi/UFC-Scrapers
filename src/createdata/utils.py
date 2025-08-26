import sys

import requests
from bs4 import BeautifulSoup


# Re-use a single ``requests`` session so we get connection pooling and can also
# configure retries in one place.  Some of the UFC endpoints occasionally return
# transient ``403``/``5xx`` responses or redirect to the HTTPS version of the
# site.  The previous implementation made a bare ``requests.get`` call without
# any headers and with redirects disabled which resulted in downloading the
# intermediate/forbidden page.  Parsing those pages produced empty or garbled
# data (missing fighter names, stats, etc.).


# ``Session`` gives us a convenient place to set a User-Agent header that more
# closely resembles a real browser and to enable automatic redirects.  This
# dramatically increases the success rate of requests to ``ufcstats.com``.
_session = requests.Session()
_session.headers.update(
    {
        # A generic but commonly accepted desktop user agent string
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
)


def make_soup(url: str) -> BeautifulSoup:
    """Return a :class:`~bs4.BeautifulSoup` object for ``url``.

    The helper now follows redirects and raises an informative error if the
    request fails.  Returning early when a non-``200`` status code is received
    prevents downstream parsing functions from silently operating on the
    "Forbidden" or "Moved" placeholder pages which previously led to completely
    incorrect data being written to ``data.csv``.
    """

    response = _session.get(url, allow_redirects=True, timeout=10)

    # ``raise_for_status`` will throw an ``HTTPError`` for 4xx/5xx responses.  We
    # intentionally let this bubble up so the caller can decide how to handle
    # failures rather than operating on bad HTML.
    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def print_progress(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    bar_length: int = 50,
) -> None:
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    percents = f"{100 * (iteration / float(total)):.2f}"
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = f'{"█" * filled_length}{"-" * (bar_length - filled_length)}'

    sys.stdout.write(f"\r{prefix} |{bar}| {percents}% {suffix}")

    if iteration == total:
        sys.stdout.write("\n")
    sys.stdout.flush()
