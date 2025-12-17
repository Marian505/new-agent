import asyncio
import base64
import os

async def get_cwd_async():
    return await asyncio.to_thread(os.getcwd)

async def is_path_exists(path: str):
    return await asyncio.to_thread(os.path.exists, path) 

async def b64encode(input_bytes):
    return await asyncio.to_thread(base64.b64encode, input_bytes)