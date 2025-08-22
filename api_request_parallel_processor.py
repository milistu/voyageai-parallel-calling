import argparse
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Generator

import aiohttp
from transformers import AutoTokenizer

SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR = 15
# 1 ms limits max throughput to 1,000 requests per second
SECONDS_TO_SLEEP_EACH_LOOP = 0.001


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ) -> None:
        """
        Calls the Voyage AI API and saves results.

        Args:
            session: aiohttp.ClientSession - the session to use to make the API call
            request_url: str - the URL of the API endpoint to call
            request_header: dict - the header to use to make the API call
            retry_queue: asyncio.Queue - the queue to use to retry failed requests
            save_filepath: str - the file to save the results to
            status_tracker: StatusTracker - the tracker to use to track the status of the request

        Returns:
            None
        """
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "rate limit" in response["error"].get("message", "").lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    conservative_factor: float,
    model_name: str,
    max_attempts: int,
    logging_level: int,
) -> None:
    """
    Processes API requests in parallel, throttling to stay under rate limits.

    Args:
        requests_filepath: str - path to the file containing the requests to be processed
        save_filepath: str - path to the file where the results will be saved
        request_url: str - URL of the API endpoint to call
        api_key: str - API key to use
        max_requests_per_minute: float - target number of requests to make per minute (will make less if limited by tokens)
        max_tokens_per_minute: float - target number of tokens to use per minute (will use less if limited by requests)
        conservative_factor: float - factor to multiply the max_requests_per_minute and max_tokens_per_minute by to start with. Voyage AI is sensitive to bursting, so start with much lower capacity.
        model_name: str - name of the model to use, this will be used to construct the tokenizer
        max_attempts: int - number of times to retry a failed request before giving up
        logging_level: int - level of logging to use; higher numbers will log fewer messages

    Returns:
        None
    """
    # Initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # Infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}

    # Initialize tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(f"voyageai/{model_name}")
    except Exception as e:
        logging.error(f"Error initializing tokenizer: {e}")
        raise e

    # Initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # Generates integer IDs of 0, 1, 2, ...
    status_tracker = (
        StatusTracker()
    )  # Single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # Initialize available capacity counts - start conservative to avoid initial burst
    # Voyage AI is sensitive to bursting, so start with much lower capacity
    available_request_capacity = max_requests_per_minute * conservative_factor
    available_token_capacity = max_tokens_per_minute * conservative_factor
    last_update_time = time.time()

    # Initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug("Initialization complete.")

    # Initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug("File opened. Entering main loop")
        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            while True:
                # Get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            # Get new request
                            request_json = json.loads(next(requests))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    tokenizer, request_json, api_endpoint
                                ),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            # If file runs out, set flag to stop reading it
                            logging.debug("Read file exhausted")
                            file_not_finished = False

                # Update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # If enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        # Update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # Call API
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None  # Reset next_request to empty

                # If all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # Main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(SECONDS_TO_SLEEP_EACH_LOOP)

                # If a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR
                ):
                    remaining_seconds_to_pause = (
                        SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logging.warning(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR)}"
                    )

        # After finishing, log final status
        logging.info(
            f"""Parallel processing complete. Results saved to {save_filepath}"""
        )
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )


def api_endpoint_from_url(request_url: str) -> str:
    """
    Extract the API endpoint from the request URL.

    Args:
        request_url: str - the URL of the API endpoint to call

    Returns:
        str - the API endpoint

    Raises:
        ValueError: if the URL doesn't match expected patterns
    """
    match = re.search(r"^https://[^/]+/v\d+/(.+)$", request_url)

    if match is None:
        raise ValueError(f"Could not extract API endpoint from URL: {request_url}")

    return match[1]


def append_to_jsonl(data: list, filename: str) -> None:
    """
    Append a json payload to the end of a jsonl file.

    Args:
        data: list - the data to append to the file
        filename: str - the file to append the data to

    Returns:
        None
    """
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    tokenizer: AutoTokenizer,
    request_json: dict,
    api_endpoint: str,
) -> int:
    """
    Count the number of tokens in the request. Only supports embedding requests.

    Args:
        tokenizer: AutoTokenizer - the tokenizer to use to count the tokens
        request_json: dict - the request to count the tokens of
        api_endpoint: str - the API endpoint to call

    Returns:
        int - the number of tokens in the request
    """
    if api_endpoint == "embeddings":
        input_data = request_json["input"]
        if isinstance(input_data, str):  # single input
            num_tokens = len(tokenizer.encode(input_data))
            return num_tokens
        elif isinstance(input_data, list):  # multiple inputs
            num_tokens = sum([len(tokenizer.encode(i)) for i in input_data])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "input" field in embedding request'
            )
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )


def task_id_generator_function() -> Generator[int]:
    """
    Generate integers 0, 1, 2, and so on.

    Returns:
        int - the next task ID
    """
    task_id = 0
    while True:
        yield task_id
        task_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_filepath")
    parser.add_argument("--save_filepath", default=None)
    parser.add_argument(
        "--request_url", default="https://api.voyageai.com/v1/embeddings"
    )
    parser.add_argument("--api_key", default=os.getenv("VOYAGE_API_KEY"))
    parser.add_argument("--max_requests_per_minute", type=int, default=2_000 * 0.5)
    parser.add_argument("--max_tokens_per_minute", type=int, default=3_000_000 * 0.5)
    parser.add_argument("--model_name", default="voyage-3-large")
    parser.add_argument("--conservative_factor", type=float, default=0.1)
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", default=logging.INFO)
    args = parser.parse_args()

    if args.save_filepath is None:
        args.save_filepath = args.requests_filepath.replace(".jsonl", "_results.jsonl")

    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=args.requests_filepath,
            save_filepath=args.save_filepath,
            request_url=args.request_url,
            api_key=args.api_key,
            max_requests_per_minute=float(args.max_requests_per_minute),
            max_tokens_per_minute=float(args.max_tokens_per_minute),
            model_name=args.model_name,
            max_attempts=int(args.max_attempts),
            logging_level=int(args.logging_level),
        )
    )
