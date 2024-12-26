# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# fmt: off

# Global imports
import re
import time
import aiofiles
import asyncio
import os
import torch
from aiobotocore.session import get_session
import bittensor as bt
from typing import List, Dict, Optional

# Local imports
from . import __version__
from .chain import ChainManager
from .config import client_config, BUCKET_SECRETS
from .logging import logger
from .schemas import Bucket

CF_REGION_NAME: str = "enam"


def get_base_url(account_id):
    """Constructs the base URL for the R2 storage endpoint."""
    return f"https://{account_id}.r2.cloudflarestorage.com"


class Comms:
    def __init__(
        self,
        wallet: "bt.wallet",
        bucket: Bucket,
        chain_manager: "ChainManager",
        save_location: str = "/tmp",
        key_prefix: str = "slice",
    ):
        """Initialize Comms handler

        Args:
            wallet: Wallet for this neuron
            bucket: Bucket configuration for this neuron
            save_location: Path for temporary storage
            key_prefix: Prefix for stored files
        """
        self.wallet = wallet
        self.bucket = bucket
        self.chain_manager = chain_manager
        self.save_location = save_location
        self.key_prefix = key_prefix
        self.session = get_session()
        self.lock = asyncio.Lock()

        # Load bucket secrets
        self.bucket_secrets = BUCKET_SECRETS

    async def put(
        self,
        state_dict: dict,
        uid: str,
        window: int,
        key: Optional[str] = None,
        global_step: int = 0,
    ):
        """
        Uploads a slice of the model parameters to the R2 bucket.

        Args:
            state_dict (dict): The state dictionary to upload.
            uid (str): Unique identifier for the upload (e.g., hotkey or user ID).
            window (int): The window number for synchronization.
            key (str, optional): Custom key for the filename. Defaults to self.key_prefix.
            global_step (int): Global training step.
        """
        key = key or self.key_prefix
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        temp_file_path = os.path.join(self.save_location, filename)
        # Include global_step in state_dict
        state_dict["global_step"] = global_step
        # Save the state_dict to a temporary file
        torch.save(state_dict, temp_file_path)

        # Upload the file to R2 bucket
        async with self.session.create_client(
            "s3",
            endpoint_url=get_base_url(self.bucket_secrets["account_id"]),
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=self.bucket_secrets["write"]["access_key_id"],
            aws_secret_access_key=self.bucket_secrets["write"]["secret_access_key"],
        ) as s3_client:
            try:
                async with aiofiles.open(temp_file_path, "rb") as f:
                    data = await f.read()
                    await s3_client.put_object(
                        Bucket=self.bucket.name, Key=filename, Body=data
                    )
                logger.debug(f"Successfully uploaded {filename} to R2 bucket.")
            except Exception as e:
                logger.error(f"Failed to upload {filename} to R2 bucket: {e}")
            finally:
                # Clean up the temporary file
                os.remove(temp_file_path)

    async def get(
        self,
        uid: str,
        window: int,
        key: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Downloads a slice of the model parameters from the R2 bucket.

        Args:
            uid (str): Unique identifier for the download.
            window (int): The window number for synchronization.
            key (str, optional): Custom key for the filename. Defaults to self.key_prefix.
            timeout (int): Timeout in seconds for the download operation.

        Returns:
            dict: The state dictionary downloaded from the bucket.
        """
        key = key or self.key_prefix
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        temp_file_path = os.path.join(self.save_location, filename)

        # Get bucket credentials for this uid
        bucket = self.chain_manager.get_bucket(uid)

        async with self.session.create_client(
            "s3",
            endpoint_url=get_base_url(bucket.account_id),
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=bucket.access_key_id,
            aws_secret_access_key=bucket.secret_access_key,
        ) as s3_client:
            try:
                # Use asyncio timeout
                async with asyncio.timeout(timeout):
                    response = await s3_client.get_object(
                        Bucket=self.bucket.name, Key=filename
                    )
                    async with aiofiles.open(temp_file_path, "wb") as f:
                        while True:
                            chunk = await response["Body"].read(
                                1024 * 1024
                            )  # Read 1 MB chunks
                            if not chunk:
                                break
                            await f.write(chunk)
                # Load the state_dict
                state_dict = torch.load(temp_file_path, map_location="cpu")
                logger.debug(f"Successfully downloaded {filename} from R2 bucket.")
                return state_dict
            except asyncio.TimeoutError:
                logger.error(f"Timeout while downloading {filename} from R2 bucket.")
                return None
            except Exception as e:
                logger.error(f"Failed to download {filename} from R2 bucket: {e}")
                return None
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

    async def get_with_retry(
        self,
        uid: str,
        window: int,
        key: Optional[str] = None,
        timeout: int = 30,
        retry_interval: float = 0.1,
    ):
        """
        Attempts to download data from the R2 bucket, retrying until success or timeout.

        Args:
            uid (str): Unique identifier for the download.
            window (int): The window number for synchronization.
            key (str, optional): Custom key for the filename.
            timeout (int): Total timeout duration for retries.
            retry_interval (float): Time to wait between retries.

        Returns:
            dict: The state dictionary downloaded from the bucket.
        """
        start_time = time.time()
        while True:
            state_dict = await self.get(uid, window, key, timeout)
            if state_dict is not None:
                return state_dict
            if time.time() - start_time > timeout:
                logger.error(f"Exceeded timeout while downloading data for UID {uid}.")
                return None
            await asyncio.sleep(retry_interval)

    async def gather(
        self,
        state_dict: Dict[str, torch.Tensor],
        my_uid: str,
        uids: List[str],
        window: int,
        key: Optional[str] = None,
        timeout: int = 30,
        device: str = "cpu",
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Gathers slices from multiple peers and assembles them for aggregation.

        Args:
            state_dict (Dict[str, torch.Tensor]): Local state dictionary.
            my_uid (str): This node's unique identifier.
            uids (List[str]): List of peer UIDs to gather data from.
            window (int): The window number for synchronization.
            key (str, optional): Custom key for filenames.
            timeout (int): Timeout for gathering data from each peer.
            device (str): Device to map tensors onto.

        Returns:
            Dict[str, List[torch.Tensor]]: Aggregated state dictionaries from all peers.
        """
        key = key or self.key_prefix
        # Put own state_dict to the bucket
        await self.put(state_dict, my_uid, window, key)

        # Gather state_dicts from peers
        gather_tasks = [
            self.get_with_retry(uid=uid, window=window, key=key, timeout=timeout)
            for uid in uids
        ]

        responses = await asyncio.gather(*gather_tasks)
        # Initialize the gather_result dictionary
        gather_result = {param_name: [] for param_name in state_dict.keys()}
        # Assemble the results
        for idx, peer_state in enumerate(responses):
            if peer_state is None:
                # Handle missing peer data, e.g., fill with zeros or skip
                for param_name in state_dict.keys():
                    gather_result[param_name].append(
                        torch.zeros_like(state_dict[param_name]).to(device)
                    )
            else:
                for param_name in state_dict.keys():
                    gather_result[param_name].append(peer_state[param_name].to(device))

        return gather_result


async def delete_old_version_files(bucket_name: str, current_version: str):
    """
    Deletes files from the S3 bucket that do not match the current version.

    Args:
        bucket_name (str): The name of the S3 bucket.
        current_version (str): The current version string.
    """
    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=get_base_url(BUCKET_SECRETS["account_id"]),
        region_name=CF_REGION_NAME,
        config=client_config,
        aws_access_key_id=BUCKET_SECRETS["write"]["access_key_id"],
        aws_secret_access_key=BUCKET_SECRETS["write"]["secret_access_key"],
    ) as s3_client:
        paginator = s3_client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket_name):
            to_delete = []
            for obj in page.get("Contents", []):
                filename = obj["Key"]
                # Check if the file version matches the current version
                match = re.match(r".+-v(.+)\.pt$", filename)
                if match:
                    file_version = match.group(1)
                    if file_version != current_version:
                        to_delete.append({"Key": filename})
                        logger.debug(f"Scheduled for deletion: {filename}")
            # Delete old versions in batches of 1000 (S3 limit for delete_objects)
            if to_delete:
                response = await s3_client.delete_objects(
                    Bucket=bucket_name, Delete={"Objects": to_delete}
                )
                deleted = response.get("Deleted", [])
                logger.info(
                    f"Deleted {len(deleted)} old version files from bucket {bucket_name}"
                )
