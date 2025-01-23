from collections import OrderedDict
from typing import Optional, List, Type, Callable, Dict

from pydantic import BaseModel, Field

from ray.data.block import UserDefinedFunction
from ray.data import Dataset
from ray.util.annotations import DeveloperAPI

from ray.llm._internal.batch.stages import (
    StatefulStage,
    wrap_preprocess,
    wrap_postprocess,
)


@DeveloperAPI
class ProcessorConfig(BaseModel):
    """The processor configuration."""

    batch_size: int = Field(
        default=64,
        description="Large batch sizes are likely to saturate the compute resources "
        "and could achieve higher throughput. On the other hand, small batch sizes "
        "are more fault-tolerant and could reduce bubbles in the data pipeline. "
        "You can tune the batch size to balance the throughput and fault-tolerance "
        "based on your use case. Default to 64.",
    )

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True


@DeveloperAPI
class Processor:
    """The processor.

    Args:
        config: The processor config.
        preprocess: Preprocess inputs to fit the processor inputs.
        postprocess: Postprocess outputs from the processor.
        accelerator_type: The accelerator type.
        concurrency: The number of concurrent requests.
    """

    # The reserved data column name. Usually we don't need to
    # change this, but if your dataset really needs to use this
    # name in your dataset and results in conflicts, you should
    # inherit the processor and customize the data_column name.
    data_column: str = "__data"

    def __init__(
        self,
        config: ProcessorConfig,
        preprocess: Optional[UserDefinedFunction] = None,
        postprocess: Optional[UserDefinedFunction] = None,
        accelerator_type: Optional[str] = None,
        concurrency: int = 1,
    ):
        self.config = config
        self.preprocess = None
        self.postprocess = None
        self.accelerator_type = accelerator_type
        self.concurrency = concurrency
        self.stages: OrderedDict[str, StatefulStage] = OrderedDict()

        if preprocess is not None:
            self.preprocess = wrap_preprocess(
                preprocess,
                self.data_column,
            )

        if postprocess is not None:
            self.postprocess = wrap_postprocess(
                postprocess,
                self.data_column,
            )

    def __call__(self, dataset: Dataset) -> Dataset:
        """Execute the processor:
        preprocess -> stages -> postprocess.
        Note that the dataset won't be materialized during the execution.

        Args:
            dataset: The input dataset.

        Returns:
            The output dataset.
        """
        if self.preprocess is not None:
            dataset = dataset.map(self.preprocess)

        for stage in self.stages.values():
            # We separate fn and fn_constructor_kwargs in Stage for better UX,
            # so we need to combine them with other map_batches_kwargs together.
            kwargs = stage.map_batches_kwargs.copy()
            kwargs["batch_size"] = self.config.batch_size
            kwargs.update({"fn_constructor_kwargs": stage.fn_constructor_kwargs})
            kwargs["fn_constructor_kwargs"]["data_column"] = self.data_column

            # Apply the stage.
            dataset = dataset.map_batches(stage.fn, **kwargs)

        if self.postprocess is not None:
            dataset = dataset.map(self.postprocess)
        return dataset

    def append_stage(self, stage: StatefulStage):
        """Append a stage before postprocess. The stage class name will be used as
        the stage name. If there are multiple stages with the same type, a suffix
        will be added to the stage name to avoid conflicts.

        Args:
            stage: The stage to append.
        """
        stage_name = type(stage).__name__

        # When a processor has multiple stages with the same type,
        # append a index suffix to the stage name to avoid conflicts.
        if stage_name in self.stages:
            num_same_type_stage = len([s for s in self.stages.values() if s is stage])
            stage_name = f"{stage_name}_{num_same_type_stage + 1}"
        self.stages[stage_name] = stage

    def list_stage_names(self) -> List[str]:
        """List the stage names of this processor in order. Preprocess and postprocess
        are not included.

        Returns:
            A list of stage names.
        """
        return list(self.stages.keys())

    def get_stage_by_name(self, name: str) -> StatefulStage:
        """Get a particular stage by its name. If the stage is not found,
        a ValueError will be raised.

        Args:
            name: The stage name.

        Returns:
            The pipeline stage.
        """
        if name in self.stages:
            return self.stages[name]
        raise ValueError(f"Stage {name} not found")


@DeveloperAPI
class ProcessorBuilder:
    """Build a processor based on the configuration."""

    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, config_type: Type[ProcessorConfig], builder: Callable):
        """A decorator to assoicate a particular pipeline config
        with its build function.
        """
        type_name = config_type.__name__
        if type_name in cls._registry:
            raise ValueError(f"Processor config type {type_name} already registered.")
        cls._registry[type_name] = builder

    @classmethod
    def build(
        cls,
        config: ProcessorConfig,
        override_stage_config_fn: Optional[Callable] = None,
        **kwargs,
    ) -> Processor:
        """Build a processor.

        Args:
            config: The processor config.
            override_stage_config_fn: Custom stages configurations.

        Returns:
            The built processor.
        """
        type_name = type(config).__name__
        if type_name not in cls._registry:
            raise ValueError(
                f"Processor config type {type_name} not registered. "
                f"Available types: {cls._registry.keys()}"
            )
        processor = cls._registry[type_name](config, **kwargs)
        if override_stage_config_fn is not None:
            for name, stage in processor.stages.items():
                override_stage_config_fn(name, stage)
        return processor
