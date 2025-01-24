from ray.llm._internal.batch.processor import (
    ProcessorConfig as _ProcessorConfig,
    Processor as _Processor,
    HttpRequestProcessorConfig as _HttpRequestProcessorConfig,
    vLLMEngineProcessorConfig as _vLLMEngineProcessorConfig,
)
from ray.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class ProcessorConfig(_ProcessorConfig):
    """The base processor configuration.

    Args:
        batch_size: Large batch sizes are likely to saturate the compute resources
            and could achieve higher throughput. On the other hand, small batch sizes
            are more fault-tolerant and could reduce bubbles in the data pipeline.
            You can tune the batch size to balance the throughput and fault-tolerance
            based on your use case.
    """

    pass


@PublicAPI(stability="alpha")
class Processor(_Processor):
    """A processor is composed of a preprocess stage, followed by one or more
    processing stages, and finally a postprocess stage. We use processor as a
    paradigm for processing data using LLMs.

    Args:
        config: The processor config.
        preprocess: A lambda function that takes a row (dict) as input and returns a
            preprocessed row (dict). The output row must contain the required fields
            for the following processing stages.
        postprocess: A lambda function that takes a row (dict) as input and returns a
            postprocessed row (dict).
        accelerator_type: The accelerator type. Default to None, meaning that only
            the CPU will be used.
        concurrency: The number of workers for data parallelism.
    """

    pass


@PublicAPI(stability="alpha")
def build_llm_processor(config: ProcessorConfig, **kwargs) -> Processor:
    """Build a LLM processor using the given config.

    Args:
        config: The processor config.
        **kwargs: Additional keyword arguments to pass to the processor.
            See `Processor` for argument details.

    Returns:
        The built processor.
    """
    from ray.llm._internal.batch.processor import ProcessorBuilder

    return ProcessorBuilder.build(config, **kwargs)


@PublicAPI(stability="alpha")
class HttpRequestProcessorConfig(_HttpRequestProcessorConfig):
    """The configuration for the HTTP request processor.

    Examples:
        .. testcode::

            import ray
            from ray.data.llm import HttpRequestProcessorConfig, build_llm_processor

            config = HttpRequestProcessorConfig(
                url="https://api.openai.com/v1/chat/completions",
                headers={"Authorization": "Bearer sk-..."},
            )
            processor = build_llm_processor(
                config,
                preprocess=lambda row: dict(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a calculator"},
                        {"role": "user", "content": f"{row['id']} ** 3 = ?"},
                    ],
                    temperature=0.3,
                    max_tokens=20,
                ),
                postprocess=lambda row: dict(
                    resp=row["choices"][0]["message"]["content"],
                ),
                concurrency=1,
            )

            ds = ray.data.range(10)
            ds = processor(ds)
            for row in ds.take_all():
                print(row)

    Args:
        url: The base URL to send the request to.
        headers: The headers to send with the request. Note that the
            default headers ("Content-Type": "application/json") is always added, so
            you don't need to specify it here.
        qps: The maximum number of requests per second to avoid rate limit.
            If None, the request will be sent sequentially.
    """

    pass


@PublicAPI(stability="alpha")
class vLLMEngineProcessorConfig(_vLLMEngineProcessorConfig):
    """TBA."""

    pass
