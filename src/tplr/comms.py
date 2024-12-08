import os
import time
import torch
import asyncio
import aiofiles
import botocore
import tempfile
from typing import List, Dict, Optional
from types import SimpleNamespace
from aiobotocore.session import get_session

import tplr as tplr

# Constants
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
BUCKET = 'decis'
CLIENT_CONFIG = botocore.config.Config(max_pool_connections=256)
LOCAL_TMP_DIR = "/tmp/local_store"

# Ensure local directory exists
os.makedirs(LOCAL_TMP_DIR, exist_ok=True)

# --------------------------- Helper Functions ---------------------------

def delete_local_directory(path: str):
    """
    Safely remove a local directory and all its contents.
    """
    if not os.path.exists(path):
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(path)


def cleanup_local_data(uid: str, current_window: int, stale_retention: int):
    """
    Clean up stale local data for a given uid.
    Deletes all directories under LOCAL_TMP_DIR/{uid}/ that have a window number 
    less than (current_window - stale_retention).
    """
    user_dir = os.path.join(LOCAL_TMP_DIR, uid)
    if not os.path.exists(user_dir):
        return

    min_allowed_window = current_window - stale_retention
    for wdir in os.listdir(user_dir):
        if wdir.isdigit():
            w = int(wdir)
            if w < min_allowed_window:
                old_path = os.path.join(user_dir, wdir)
                tplr.logger.debug(f"Removing stale local directory: {old_path}")
                try:
                    delete_local_directory(old_path)
                except Exception as e:
                    tplr.logger.debug(f"Error removing stale directory {old_path}: {e}")


async def cleanup_s3_data(uid: str, current_window: int, stale_retention: int):
    """
    Clean up stale S3 data for a given uid.
    Deletes all objects in S3 for windows less than (current_window - stale_retention).
    Keys are assumed in the format: "uid/window/key".
    """
    min_allowed_window = current_window - stale_retention
    prefix = f"{uid}/"

    session = get_session()
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=CLIENT_CONFIG,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ) as s3_client:

        continuation_token = None

        while True:
            list_args = {
                "Bucket": BUCKET,
                "Prefix": prefix,
                "MaxKeys": 1000
            }
            if continuation_token:
                list_args["ContinuationToken"] = continuation_token

            response = await s3_client.list_objects_v2(**list_args)
            contents = response.get("Contents", [])

            # Identify stale objects to delete
            stale_objects = []
            for obj in contents:
                key = obj["Key"]
                # Key format: uid/window/key
                parts = key.split("/")
                if len(parts) < 2:
                    continue
                try:
                    w = int(parts[1])
                except ValueError:
                    continue

                if w < min_allowed_window:
                    stale_objects.append({"Key": key})

            # Batch delete stale objects
            if stale_objects:
                tplr.logger.debug(f"Removing stale S3 objects for {uid}: {stale_objects}")
                await s3_client.delete_objects(
                    Bucket=BUCKET,
                    Delete={"Objects": stale_objects}
                )

            if response.get("IsTruncated"):
                continuation_token = response.get("NextContinuationToken")
            else:
                break


async def s3_put_object(key: str, data: bytes):
    """
    Upload object to S3 at the specified key.
    """
    session = get_session()
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=CLIENT_CONFIG,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ) as s3_client:
        await s3_client.put_object(Bucket=BUCKET, Key=key, Body=data)


async def s3_get_object(key: str, timeout: int) -> Optional[dict]:
    """
    Download an object from S3 and return the loaded state_dict.
    Returns None if not found or on error.
    """
    session = get_session()
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=CLIENT_CONFIG,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ) as s3_client:
        try:
            response = await asyncio.wait_for(
                s3_client.get_object(Bucket=BUCKET, Key=key),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            tplr.logger.debug(f"Timeout occurred while downloading {key}.")
            return None
        except Exception as e:
            tplr.logger.debug(f"An error occurred during GET {key}: {e}")
            return None

        # Save to a temporary file and load
        with tempfile.NamedTemporaryFile(delete=True, suffix='.pt') as temp_file:
            temp_file_path = temp_file.name
            async with aiofiles.open(temp_file_path, "wb") as outfile:
                while True:
                    chunk = await response["Body"].read(1 * 1024 * 1024)
                    if not chunk:
                        break
                    await outfile.write(chunk)

            # Load the object
            try:
                with open(temp_file_path, 'rb') as f:
                    state_dict = torch.load(f, weights_only=True)
                return state_dict
            except Exception as e:
                tplr.logger.debug(f"Error loading state_dict from {key}: {e}")
                return None


# --------------------------- Main Functions ---------------------------

async def put(
    state_dict: dict, 
    uid: str, 
    window: int, 
    key: str, 
    local: bool = True, 
    stale_retention: int = 10
):
    """
    PUT operation: Store the state_dict either locally or in S3.
    Also cleans up stale data.

    Args:
        state_dict (dict): A dictionary of tensor parameters.
        uid (str): Unique identifier.
        window (int): Window index.
        key (str): Key name.
        local (bool): If True, store locally, else store in S3.
        stale_retention (int): Number of recent windows to keep.
    """
    tplr.logger.debug(f"PUT {uid}/{window}/{key} -->")

    # Save state_dict to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_file:
        torch.save(state_dict, temp_file)
        temp_file_path = temp_file.name

    try:
        if local:
            # Cleanup old local data before putting the new file
            cleanup_local_data(uid=uid, current_window=window, stale_retention=stale_retention)

            # Store locally
            local_dir = os.path.join(LOCAL_TMP_DIR, uid, str(window))
            os.makedirs(local_dir, exist_ok=True)
            final_path = os.path.join(local_dir, f"{key}.pt")
            os.replace(temp_file_path, final_path)
        else:
            # Cleanup old S3 data before putting the new file
            await cleanup_s3_data(uid=uid, current_window=window, stale_retention=stale_retention)

            # Upload to S3
            object_key = f"{uid}/{window}/{key}"
            async with aiofiles.open(temp_file_path, "rb") as f:
                data = await f.read()
            await s3_put_object(object_key, data)

            # Remove temporary file after successful upload
            os.remove(temp_file_path)

    except Exception as e:
        tplr.logger.debug(f"PUT error {uid}/{window}/{key}: {e}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    tplr.logger.debug(f"PUT {uid}/{window}/{key} <--")


async def get(
    uid: int, 
    window: int, 
    key: str, 
    timeout: int = 30, 
    local: bool = True, 
    stale_retention: int = 10
) -> Optional[dict]:
    """
    GET operation: Retrieve state_dict from local or S3 storage.
    Also cleans up stale data before retrieving.

    Args:
        uid (int): Unique identifier.
        window (int): Window index.
        key (str): Key name.
        timeout (int): Timeout in seconds for retrieval.
        local (bool): If True, read from local storage. If False, from S3.
        stale_retention (int): Number of recent windows to keep.

    Returns:
        dict or None: The loaded state_dict, or None on failure.
    """
    full_key = f"{uid}/{window}/{key}"
    tplr.logger.debug(f"GET {full_key} -->")

    if local:
        # Cleanup old local data before reading
        cleanup_local_data(uid=str(uid), current_window=window, stale_retention=stale_retention)

        # Local file path
        local_path = os.path.join(LOCAL_TMP_DIR, str(uid), str(window), f"{key}.pt")
        if not os.path.exists(local_path):
            tplr.logger.debug(f"Local file not found: {local_path}")
            return None
        try:
            state_dict = torch.load(local_path, weights_only=True)
        except Exception as e:
            tplr.logger.debug(f"Error loading local file {local_path}: {e}")
            return None
        tplr.logger.debug(f"GET {full_key} <--")
        return state_dict
    else:
        # Cleanup old S3 data before reading
        await cleanup_s3_data(uid=str(uid), current_window=window, stale_retention=stale_retention)

        # Download from S3
        state_dict = await s3_get_object(full_key, timeout)
        tplr.logger.debug(f"GET {full_key} <--")
        return state_dict


async def get_with_retry(
    uid: int, 
    window: int, 
    key: str, 
    timeout: int, 
    local: bool = True, 
    stale_retention: int = 10
) -> Optional[dict]:
    """
    Retry GET until success or timeout.

    Args:
        uid (int): Unique identifier.
        window (int): Window index.
        key (str): Key name.
        timeout (int): Overall timeout in seconds.
        local (bool): If True, use local filesystem. Otherwise S3.
        stale_retention (int): Number of recent windows to keep.

    Returns:
        dict or None: The loaded state_dict or None if timed out.
    """
    start_time = time.time()
    end_time = start_time + timeout

    while True:
        if time.time() >= end_time:
            tplr.logger.debug(f"GET {uid}/{window}/{key} timed out.")
            return None

        state_dict = await get(uid=uid, window=window, key=key, local=local, stale_retention=stale_retention)
        if state_dict is not None:
            return state_dict

        # Retry after a short delay
        await asyncio.sleep(0.1)


async def gather(
    state_dict: Dict[str, torch.Tensor], 
    my_uid: int, 
    uids: List[int], 
    window: int, 
    key: str, 
    timeout: int,
    device: str,
    local: bool = True,
    stale_retention: int = 10
) -> SimpleNamespace:
    """
    Gather results from multiple peers. Optionally store own state_dict first, 
    then retrieve others.

    Args:
        state_dict (dict): The local state_dict to put (can be None).
        my_uid (int): The UID of the local participant.
        uids (list): The UIDs of other participants.
        window (int): Window index.
        key (str): Key name.
        timeout (int): Timeout for retrieval operations.
        device (str): Device to place the loaded tensors.
        local (bool): If True, use local file system. Otherwise S3.
        stale_retention (int): Number of recent windows to keep.

    Returns:
        SimpleNamespace: {time, upload_bytes, download_bytes, success_rate, successes, state_dict}
    """
    start_time = time.time()

    # Upload metrics
    upload_bytes = 0
    download_bytes = 0
    successes = []

    # Put the local state_dict if available
    if state_dict is not None:
        await put(
            state_dict=state_dict,
            uid=str(my_uid),
            window=window,
            key=key,
            local=local,
            stale_retention=stale_retention
        )
        upload_bytes = sum(tensor.element_size() * tensor.nelement() for tensor in state_dict.values())

    # Prepare tasks to get state_dicts from other UIDs
    gather_tasks = [
        get_with_retry(
            uid=uid,
            window=window,
            key=key,
            timeout=timeout,
            local=local,
            stale_retention=stale_retention
        )
        for uid in uids
    ]

    # If we have no local state_dict, we will initialize the structure after we get at least one response.
    if state_dict is None:
        gather_result = {}
    else:
        gather_result = {
            k: [torch.zeros_like(v).to(device) for _ in uids]
            for k, v in state_dict.items()
        }

    # Await all responses
    responses = await asyncio.gather(*gather_tasks)

    # Process responses
    for idx, resp in enumerate(responses):
        if resp is None:
            successes.append(False)
            continue

        successes.append(True)

        # If we didn't have a local state_dict and this is the first successful response,
        # initialize gather_result now.
        if not gather_result and resp is not None:
            gather_result = {
                k: [torch.zeros_like(v).to(device) for _ in uids]
                for k, v in resp.items()
            }

        # Fill in data from this response
        for k, tensor in resp.items():
            gather_result[k][idx] = tensor.to(device)
            download_bytes += tensor.element_size() * tensor.nelement()

    # Calculate success metrics
    success_rate = sum(successes) / len(successes) if successes else 0
    total_time = time.time() - start_time

    return SimpleNamespace(
        time=total_time,
        upload_bytes=upload_bytes,
        download_bytes=download_bytes,
        success_rate=success_rate,
        successes=successes,
        state_dict=gather_result
    )