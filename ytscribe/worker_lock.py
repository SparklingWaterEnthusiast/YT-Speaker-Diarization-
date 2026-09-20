"""OS-owned processing lease: released even if the process crashes."""
import os
from contextlib import contextmanager, ExitStack
from pathlib import Path


@contextmanager
def processing_lease(cache_dir: Path, database_dir: Path | None = None):
    # Lock both ownership domains: changing cache settings in a second app
    # must not allow recovery/mutation of a database with an active worker.
    roots = {Path(cache_dir).resolve()}
    if database_dir is not None:
        roots.add(Path(database_dir).resolve())
    with ExitStack() as stack:
        for root in sorted(roots, key=str):
            stack.enter_context(_directory_lease(root))
        yield


@contextmanager
def _directory_lease(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / '.worker.lock'
    handle = path.open('a+b')
    if os.fstat(handle.fileno()).st_size == 0:
        handle.write(b'0'); handle.flush()
    handle.seek(0)
    locked = False
    try:
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise RuntimeError('Another YTScribe worker is processing this cache. '
                               'Stop it before starting another run.') from exc
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
