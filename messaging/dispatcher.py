import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=5)


def dispatch(task_func, *args, **kwargs):
    """
    Celery-like dispatcher.
    Today: ThreadPoolExecutor
    Tomorrow: Celery.delay()
    """
    logger.debug(
        "Dispatching task",
        extra={
            "task": task_func.__name__,
            "args": args,
            "kwargs": kwargs,
        },
    )

    _executor.submit(task_func, *args, **kwargs)
