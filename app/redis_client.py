import os
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_DB = os.getenv("REDIS_DB", 0)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "super_password")
redis_client = redis.Redis(
    host=REDIS_HOST, port=int(REDIS_PORT), db=int(REDIS_DB), password=REDIS_PASSWORD
)

# Test Redis connection
try:
    redis_client.ping()
    print("Connected to Redis!")
except Exception as e:
    print(f"Redis connection error: {e}")
