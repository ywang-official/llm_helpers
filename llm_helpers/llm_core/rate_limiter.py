import asyncio
from typing import Optional, Dict, Any
import time
from collections import deque
import logging

logger = logging.getLogger(__name__)

class LLMRateLimiter:
    def __init__(self, max_concurrent_sessions: int = 5):
        """
        Initialize the rate limiter with a maximum number of concurrent sessions.
        
        Args:
            max_concurrent_sessions (int): Maximum number of concurrent LLM sessions allowed
        """
        self.max_concurrent_sessions = max_concurrent_sessions
        self.semaphore = asyncio.Semaphore(max_concurrent_sessions)
        self.waiting_queue = deque()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def acquire_session(self, session_id: str) -> None:
        """
        Acquire a session slot. If no slots are available, wait in the queue.
        
        Args:
            session_id (str): Unique identifier for the session
        """
        future = None
        async with self._lock:
            if session_id in self.active_sessions:
                logger.warning(f"Session {session_id} already exists")
                return
            
            # Add to waiting queue if no slots available
            if len(self.active_sessions) >= self.max_concurrent_sessions:
                logger.info(f"Session {session_id} waiting for slot")
                future = asyncio.Future()
                self.waiting_queue.append((session_id, future))

        if future:
            await future  # Wait until a slot becomes available
            
        # Acquire the semaphore
        await self.semaphore.acquire()
        
        # Record the session
        async with self._lock:
            self.active_sessions[session_id] = {
                'start_time': time.time(),
                'status': 'active'
            }
            logger.info(f"Session {session_id} started")
    
    async def release_session(self, session_id: str) -> None:
        """
        Release a session slot and notify the next waiting session if any.
        
        Args:
            session_id (str): Unique identifier for the session to release
        """
        async with self._lock:
            if session_id not in self.active_sessions:
                logger.warning(f"Session {session_id} not found")
                return
            
            # Remove the session
            del self.active_sessions[session_id]
            self.semaphore.release()
            logger.info(f"Session {session_id} released")
            
            # Notify the next waiting session if any
            if self.waiting_queue:
                next_session_id, future = self.waiting_queue.popleft()
                future.set_result(None)
                logger.info(f"Notified waiting session {next_session_id}")
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific session.
        
        Args:
            session_id (str): Unique identifier for the session
            
        Returns:
            Optional[Dict[str, Any]]: Session status information or None if not found
        """
        async with self._lock:
            return self.active_sessions.get(session_id)
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get the current status of the rate limiter including active sessions and waiting queue.
        
        Returns:
            Dict[str, Any]: Current status of the rate limiter
        """
        async with self._lock:
            return {
                'active_sessions': len(self.active_sessions),
                'waiting_sessions': len(self.waiting_queue),
                'max_concurrent_sessions': self.max_concurrent_sessions,
                'active_session_ids': list(self.active_sessions.keys()),
                'waiting_session_ids': [sid for sid, _ in self.waiting_queue]
            }

# Backward compatibility - create a default global instance
# This will be deprecated in favor of instance-based rate limiters
_default_rate_limiter = None

def get_default_rate_limiter() -> LLMRateLimiter:
    """
    Get the default global rate limiter instance.
    
    This is kept for backward compatibility but is deprecated.
    Prefer using instance-based rate limiters through LLM handler configuration.
    
    Returns:
        LLMRateLimiter: Default global rate limiter instance
    """
    global _default_rate_limiter
    if _default_rate_limiter is None:
        _default_rate_limiter = LLMRateLimiter()
    return _default_rate_limiter

# Backward compatibility alias - deprecated
rate_limiter = get_default_rate_limiter()