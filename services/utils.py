import asyncio, functools, inspect, time

def retry(times=3):
    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                for attempt in range(times):
                    try:
                        return await func(*args, **kwargs)
                    except Exception:
                        if attempt == times - 1:
                            raise
                        await asyncio.sleep(2 ** attempt)
            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == times - 1:
                        raise
                    time.sleep(2 ** attempt)
        return wrapper
    return decorator
