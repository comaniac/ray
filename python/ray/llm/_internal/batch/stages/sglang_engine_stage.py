"""The stage that runs SGLang engine."""
import asyncio
import importlib
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass
from pydantic import root_validator
from typing import TYPE_CHECKING, Any, Dict, AsyncIterator, Optional, List, Tuple

from ray.llm._internal.batch.stages.base import (
    StatefulStage,
    StatefulStageUDF,
)

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class SGLangEngineWrapper:
    """Wrapper around the SGLang engine to handle async requests.
    Args:
        *args: The positional arguments for the engine.
        max_pending_requests: The maximum number of pending requests in the queue.
        runtime_env: The runtime environment to use for the SGLang engine.
        **kwargs: The keyword arguments for the engine.
    """

    @dataclass(frozen=True)
    class LLMRequest:
        """A request to the SGLang wrapper."""

        # The request ID for the SGLang engine (unique per replica).
        request_id: int
        # The full prompt string (with chat template applied if any).
        prompt: str
        # The images inputs for the multimodal model.
        images: List["Image.Image"]
        # The tokenized prompt IDs. If None, then the string prompt will be
        # tokenized by the LLM engine. This is not recommended for performance reasons.
        prompt_token_ids: Optional[List[int]]
        # The sampling or pooling parameters. Use Any to avoid importing vLLM.
        params: Any

    def __init__(
        self,
        *args,
        max_pending_requests: int = -1,
        runtime_env: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.request_id = 0
        self.task_type = kwargs.get("task", "generate")
        if self.task_type != "generate":
            raise ValueError("SGLang engine only supports generate task.")

        # Setup os.environ before importing SGLang to make sure the environment
        # variables are effective
        if runtime_env is not None and "env_vars" in runtime_env:
            os.environ.update(runtime_env["env_vars"])

        # Lazy import SGLang here.
        self.sgl = importlib.import_module("sglang")

        logger.info(
            "Initializing SGLang engine with args=%s, kwargs=%s",
            ", ".join(str(arg) for arg in args),
            ", ".join(f"{k}={v}" for k, v in kwargs.items()),
        )
        self.engine = self.sgl.Engine(
            *args,
            **kwargs,
        )

        # Performance gets really bad if there are too many requests in the pending queue.
        # We work around it by introducing another queue that gates how many requests we are
        # sending to the engine at once.
        # This is not a queue of requests. Instead, this queue holds "slots". Each time
        # we add a new request, we take one slot. When a request finishes, we add a new
        # slot.
        self.max_pending_requests = max_pending_requests
        self.free_queue: asyncio.Queue[bool] = asyncio.Queue()
        if self.max_pending_requests > 0:
            for _ in range(self.max_pending_requests):
                self.free_queue.put_nowait(True)

    def _prepare_llm_request(self, batch: Dict[str, Any]) -> List[LLMRequest]:
        """Prepare the inputs for LLM inference.

        Args:
            batch: The batch.

        Returns:
            A list of LLMRequest.
        """
        # We set skip_tokenizer_init=True to disable the tokenizer,
        # so the SGLang engine can only accept tokenized input.
        tokenized_prompt = batch.pop("tokenized_prompt")
        params = batch.pop("sampling_params")

        if "prompt" in batch:
            prompt = batch.pop("prompt")
        else:
            prompt = None

        request = self.LLMRequest(
            request_id=self.request_id,
            prompt=prompt,
            prompt_token_ids=tokenized_prompt,
            images=[],
            params=params,
        )
        self.request_id += 1
        return request

    def _parse_llm_output(
        self, request: LLMRequest, output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse the LLM output.

        Args:
            output: The LLM output.

        Returns:
            The parsed output.
        """
        if not isinstance(output, dict):
            logger.error(output.body)

        output_data = {
            "prompt": request.prompt,
            "num_input_tokens": output["meta_info"]["prompt_tokens"],
            "num_generated_tokens": output["meta_info"]["completion_tokens"],
            "generated_tokens": output["token_ids"],
        }
        return output_data

    async def generate_async(
        self, row: Dict[str, Any]
    ) -> Tuple[LLMRequest, Dict[str, Any]]:
        """Process a single request.

        Args:
            request: The request.

        Returns:
            A tuple of index in batch, request output and bypassed custom fields.
        """
        request = self._prepare_llm_request(row)

        # If free queue is used, guard the request here until a slot is available.
        if self.max_pending_requests > 0:
            await self.free_queue.get()

        output = await self.engine.async_generate(
            input_ids=request.prompt_token_ids.tolist(),
            sampling_params=request.params,
        )

        # If free queue is used, release the slot.
        if self.max_pending_requests > 0:
            self.free_queue.put_nowait(True)

        return request, self._parse_llm_output(request, output)

    def shutdown(self):
        """Shutdown the SGLang engine. This kills child processes forked
        by the SGLang engine. If not called, the child processes will be
        orphaned and will not be killed when the parent process exits.
        """
        if hasattr(self.engine, "shutdown"):
            logger.info("Shutting down SGLang engine")
            self.engine.shutdown()


class SGLangEngineStageUDF(StatefulStageUDF):
    def __init__(
        self,
        data_column: str,
        model: str,
        engine_kwargs: Dict[str, Any],
        task_type: str = "generate",
        runtime_env: Optional[Dict[str, Any]] = None,
        max_pending_requests: Optional[int] = None,
    ):
        """
        Initialize the HttpRequestUDF.
        Args:
            data_column: The data column name.
            engine_kwargs: The kwargs to pass to the SGLang engine.
            task_type: The task to use for the SGLang engine (e.g., "generate", "embed", etc).
            runtime_env: The runtime environment to use for the SGLang engine.
            max_pending_requests: The maximum number of pending requests. If None,
                it will be set to 1.1 * max_running_requests.
        """
        super().__init__(data_column)
        self.model = model

        # Setup runtime env.
        self.runtime_env = runtime_env or {}

        # Setup SGLang engine kwargs.
        self.engine_kwargs = self.normalize_engine_kwargs(engine_kwargs)

        # Set up the max pending requests.
        self.max_pending_requests = max_pending_requests or -1
        if (
            self.max_pending_requests == -1
            and "max_running_requests" in self.engine_kwargs
        ):
            self.max_pending_requests = math.ceil(
                self.engine_kwargs["max_running_requests"] * 1.1
            )
        if self.max_pending_requests == -1:
            logger.info("Max pending requests is set to %d", self.max_pending_requests)

        # Create an LLM engine.
        self.llm = SGLangEngineWrapper(
            model_path=model,
            max_pending_requests=self.max_pending_requests,
            runtime_env=self.runtime_env,
            **self.engine_kwargs,
        )

    def normalize_engine_kwargs(
        self,
        engine_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalize the engine kwargs.

        Args:
            engine_kwargs: The kwargs to normalize.

        Returns:
            The normalized kwargs.
        """
        # Remove model from engine kwargs if set.
        model_path = engine_kwargs.pop("model_path", None)
        if model_path is not None and model_path != self.model:
            logger.warning(
                "The model_path set in engine kwargs (%s) is different from the "
                "stage (%s). Please remove 'model_path' from engine kwargs.",
                model_path,
                self.model,
            )

        skip_tokenizer_init = engine_kwargs.get("skip_tokenizer_init", True)
        if not skip_tokenizer_init:
            logger.warning("Force skip_tokenizer_init=True for SGLang engine")
        engine_kwargs["skip_tokenizer_init"] = True

        return engine_kwargs

    async def udf(self, batch: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        """
        Run the vLLM engine.
        Args:
            batch: A list of rows to run the vLLM engine on.
        Returns:
            The response of the vLLM engine.
        """
        batch_uuid = uuid.uuid4()
        t = time.perf_counter()

        tasks = [asyncio.create_task(self.llm.generate_async(row)) for row in batch]

        idx = 0
        time_taken = -1.0
        for resp in asyncio.as_completed(tasks):
            request, output = await resp
            time_taken = time.perf_counter() - t

            yield {
                **output,
                "request_id": request.request_id,
                "batch_uuid": batch_uuid.hex,
                "index_in_batch": idx,
                "time_taken_llm": time_taken,
                "params": str(request.params),
            }
            idx += 1

        logger.info(
            "[SGLang] Elapsed time for batch %s with size %d: %s",
            batch_uuid.hex,
            len(batch),
            time_taken,
        )

    @property
    def expected_input_keys(self) -> List[str]:
        """The expected input keys."""
        return ["tokenized_prompt", "sampling_params"]

    def __del__(self):
        if hasattr(self, "llm"):
            self.llm.shutdown()


class SGLangEngineStage(StatefulStage):
    """
    A stage that runs SGLang engine.
    """

    fn: StatefulStageUDF = SGLangEngineStageUDF
    fn_constructor_kwargs: Dict[str, Any]
    map_batches_kwargs: Dict[str, Any] = dict(
        concurrency=1,
    )

    @root_validator(pre=True)
    def post_init(cls, values):
        """Post-initialize the stage. Specifically,
        this function determines the num_gpus and Ray remote args
        for the .map_batches() call in this stage.

        Args:
            values: The raw stage values.
        Returns:
            The updated values.
        """
        map_batches_kwargs = values["map_batches_kwargs"]
        accelerator_type = map_batches_kwargs.get("accelerator_type", "")
        fn_constructor_kwargs = values["fn_constructor_kwargs"]
        runtime_env = fn_constructor_kwargs.get("runtime_env", {})
        engine_kwargs = fn_constructor_kwargs.get("engine_kwargs", {})

        ray_remote_args = {"runtime_env": runtime_env}
        if accelerator_type:
            ray_remote_args["accelerator_type"] = accelerator_type

        # Setup num_gpus required per vLLM engine.
        num_gpus = engine_kwargs.get("tensor_parallel_size", 1)

        # For TP only case, we use multi-processing engines, so we only
        # need to set the correct num_gpus, which lets Ray allocate the
        # number of GPUs, and the engine will spawn the number of processes
        # to use them.
        map_batches_kwargs["num_gpus"] = num_gpus
        map_batches_kwargs.update(ray_remote_args)
        return values
