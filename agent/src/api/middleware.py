"""
Middleware for the LangGraph Programming Agent API.
Provides request logging, metrics collection, and monitoring.
"""

import time
import logging
from typing import Callable
from datetime import datetime
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging incoming HTTP requests.
    
    Logs request details including:
    - Method and path
    - Client IP
    - Response status
    - Request duration
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and log details.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/endpoint in chain
        
        Returns:
            HTTP response
        """
        
        # Record start time
        start_time = time.time()
        
        # Extract request details
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        
        # Log incoming request
        logger.info(f"→ {method} {path} from {client_ip}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"← {method} {path} → {response.status_code} "
                f"({duration:.3f}s)"
            )
            
            # Add custom headers
            response.headers["X-Request-ID"] = str(int(start_time * 1000000))
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
        
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            logger.error(
                f"✗ {method} {path} failed after {duration:.3f}s: {str(e)}",
                exc_info=True
            )
            raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting request metrics.
    
    Tracks:
    - Request count per endpoint
    - Average response time per endpoint
    - Error rates
    - Total requests
    """
    
    def __init__(self, app):
        """Initialize metrics middleware."""
        super().__init__(app)
        
        # Metrics storage
        self.metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "endpoint_counts": defaultdict(int),
            "endpoint_durations": defaultdict(list),
            "endpoint_errors": defaultdict(int),
            "status_codes": defaultdict(int),
            "start_time": datetime.now()
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and collect metrics.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/endpoint in chain
        
        Returns:
            HTTP response
        """
        
        # Record start time
        start_time = time.time()
        
        # Extract request details
        method = request.method
        path = request.url.path
        endpoint = f"{method} {path}"
        
        # Increment request count
        self.metrics["total_requests"] += 1
        self.metrics["endpoint_counts"][endpoint] += 1
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            self.metrics["endpoint_durations"][endpoint].append(duration)
            self.metrics["status_codes"][response.status_code] += 1
            
            # Track errors (4xx and 5xx)
            if response.status_code >= 400:
                self.metrics["total_errors"] += 1
                self.metrics["endpoint_errors"][endpoint] += 1
            
            # Add metrics headers
            response.headers["X-Total-Requests"] = str(self.metrics["total_requests"])
            response.headers["X-Endpoint-Count"] = str(self.metrics["endpoint_counts"][endpoint])
            
            return response
        
        except Exception as e:
            # Record error
            duration = time.time() - start_time
            self.metrics["total_errors"] += 1
            self.metrics["endpoint_errors"][endpoint] += 1
            self.metrics["endpoint_durations"][endpoint].append(duration)
            self.metrics["status_codes"][500] += 1
            
            raise
    
    def get_metrics(self) -> dict:
        """
        Get current metrics.
        
        Returns:
            Dictionary containing all collected metrics
        """
        
        # Calculate average durations
        avg_durations = {}
        for endpoint, durations in self.metrics["endpoint_durations"].items():
            if durations:
                avg_durations[endpoint] = sum(durations) / len(durations)
        
        # Calculate uptime
        uptime = datetime.now() - self.metrics["start_time"]
        
        return {
            "total_requests": self.metrics["total_requests"],
            "total_errors": self.metrics["total_errors"],
            "error_rate": (
                self.metrics["total_errors"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0
                else 0.0
            ),
            "uptime_seconds": uptime.total_seconds(),
            "endpoint_counts": dict(self.metrics["endpoint_counts"]),
            "endpoint_errors": dict(self.metrics["endpoint_errors"]),
            "average_durations": avg_durations,
            "status_codes": dict(self.metrics["status_codes"]),
            "start_time": self.metrics["start_time"].isoformat()
        }
    
    def reset_metrics(self):
        """Reset all metrics."""
        
        self.metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "endpoint_counts": defaultdict(int),
            "endpoint_durations": defaultdict(list),
            "endpoint_errors": defaultdict(int),
            "status_codes": defaultdict(int),
            "start_time": datetime.now()
        }
