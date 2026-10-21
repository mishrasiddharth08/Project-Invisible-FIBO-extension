"""Identity of reviewed official repacks; filenames are never proof of identity."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

PINS = json.loads((Path(__file__).with_name('official_assets.json')).read_text())


@lru_cache(maxsize=16)
def _identify(path, size, modified_ns, role, file_identity=None):
    candidates = [entry for entry in PINS['assets'] if entry['bytes'] == size and entry['role'] == role]
    if not candidates:
        raise ValueError(f'Unrecognized official {role} asset size: {path}')
    with open(path, 'rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    for entry in candidates:
        if entry['sha256'] == digest:
            return dict(entry)
    raise ValueError(f'Official {role} SHA256 does not match reviewed assets: {path}')


def identify(path, role):
    path = Path(path).resolve()
    stat = path.stat()
    identity=(stat.st_dev,stat.st_ino,stat.st_ctime_ns)
    # Windows can coalesce last-write timestamps across rapid tiny rewrites.
    # Small inputs are cheap to hash every time; replacing a large model also
    # invalidates the cache through its file identity, not just its filename.
    method=_identify.__wrapped__ if stat.st_size<1024**2 else _identify
    return method(str(path),stat.st_size,stat.st_mtime_ns,role,identity)
