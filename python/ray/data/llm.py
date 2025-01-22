from ray.llm._internal.batch.processor import (
    ProcessorConfig as _ProcessorConfig,
    Processor as _Processor,
    HttpRequestProcessorConfig as _HttpRequestProcessorConfig,
)
from ray.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class ProcessorConfig(_ProcessorConfig):
    """The base processor configuration."""

    pass


@PublicAPI(stability="alpha")
class Processor(_Processor):
    """A processor is composed of a preprocess stage, followed by one or more
    processing stages, and finally a postprocess stage. We use processor as a
    paradigm for processing data using LLMs.
    """

    pass


@PublicAPI(stability="alpha")
def build_llm_processor(config: ProcessorConfig, **kwargs) -> Processor:
    """Build a LLM processor using the given config.

    Args:
        config: The processor config.
        **kwargs: Additional keyword arguments to pass to the processor builder.
            - preprocess: The user defined function (UDF) to preprocess the input.
                The UDF should take a row (dict) as input and return a preprocessed
                row (dict). The output row must contain the required fields for the
                following processing stages.
            - postprocess: The user defined function (UDF) to postprocess the output.
                The UDF should take a row (dict) as input and return a postprocessed
                row (dict).
            - accelerator_type: The accelerator type.
            - concurrency: The number of concurrent core engines in this processor.

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
        headers: The headers to send with the request. Note that the default
            headers ("Content-Type": "application/json") is always added, so
            you don't need to specify it here.
        qps: The maximum number of requests per second to avoid rate limit.
            If None, the request will be sent sequentially.
    """

    pass
