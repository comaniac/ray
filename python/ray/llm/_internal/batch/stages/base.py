"""TBA"""
import logging
from typing import Any, Dict, AsyncIterator, List, Callable

import pyarrow
from pydantic import BaseModel, Field
from ray.data.block import UserDefinedFunction

logger = logging.getLogger(__name__)


def wrap_preprocess(
    fn: UserDefinedFunction,
    processor_data_column: str,
) -> Callable:
    """Wrap the preprocess function, so that the output schema of the
    preprocess is normalized to {processor_data_column: fn(row), other input columns}.

    Args:
        fn: The function to be applied.
        processor_data_column: The internal data column name of the processor.

    Returns:
        The wrapped function.
    """

    def _preprocess(row: dict[str, Any]) -> dict[str, Any]:
        # First put everything into processor_data_column.
        outputs = {processor_data_column: row}

        # Then apply the preprocess function and add its outputs.
        preprocess_output = fn(row)
        outputs[processor_data_column].update(preprocess_output)
        return outputs

    return _preprocess


def wrap_postprocess(
    fn: UserDefinedFunction,
    processor_data_column: str,
) -> Callable:
    """Wrap the postprocess function to remove the processor_data_column.
    Note that we fuly rely on users to determine which columns to carry over.

    Args:
        fn: The function to be applied.
        processor_data_column: The internal data column name of the processor.

    Returns:
        The wrapped function.
    """

    def _postprocess(row: dict[str, Any]) -> dict[str, Any]:
        if processor_data_column not in row:
            raise ValueError(
                f"[Internal] {processor_data_column} not found in row {row}"
            )

        return fn(row[processor_data_column])

    return _postprocess


class StatefulStageUDF:
    def __init__(self, data_column: str):
        self.data_column = data_column

    async def __call__(self, batch: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """A stage UDF wrapper that processes the input and output columns
        before and after the UDF. The expected schema of "batch" is:
        {data_column: {
            dataset columns,
            other intermediate columns
         },
         ...other metadata columns...,
        }.

        The input of the UDF will then [dataset columns and other intermediate columns].
        In addition, while the output of the UDF depends on the UDF implementation,
        the output schema is expected to be
        {data_column: {
            dataset columns,
            other intermediate columns,
            UDF output columns (will override above columns if they have the same name)
         },
         ...other metadata columns...,
        }.
        And this will become the input of the next stage.

        Args:
            batch: The input batch.

        Returns:
            An async iterator of the outputs.
        """
        # Handle the case where the batch is empty.
        # FIXME: This should not happen.
        if isinstance(batch, pyarrow.lib.Table) and batch.num_rows > 0:
            yield {}
            return

        if self.data_column not in batch:
            raise ValueError(
                f"[Internal] {self.data_column} not found in batch {batch}"
            )

        inputs = batch.pop(self.data_column)
        if hasattr(inputs, "tolist"):
            inputs = inputs.tolist()
        self.validate_inputs(inputs)

        idx = 0
        async for output in self.udf(inputs):
            # Add stage outputs to the data column of the row.
            inputs[idx].update(output)
            yield {self.data_column: [inputs[idx]]}
            idx += 1

    def validate_inputs(self, inputs: List[Dict[str, Any]]):
        """Validate the inputs to make sure the required keys are present.

        Args:
            inputs: The inputs.

        Raises:
            ValueError: If the required keys are not found.
        """
        expected_input_keys = self.expected_input_keys
        if not expected_input_keys:
            return

        input_keys = set(inputs[0].keys())
        missing_required = set(expected_input_keys) - input_keys
        if missing_required:
            raise ValueError(
                f"Required input keys {missing_required} not found at the input of "
                f"{self.__class__.__name__}. Input keys: {input_keys}"
            )

    @property
    def expected_input_keys(self) -> List[str]:
        """A list of required input keys. Missing required keys will raise
        an exception.

        Returns:
            A list of required input keys.
        """
        return []

    async def udf(self, rows: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        raise NotImplementedError("StageUDF must implement the udf method")


class StatefulStage(BaseModel):
    """
    A basic building block to compose a Processor.
    """

    fn: StatefulStageUDF = Field(
        description="The well-optimized stateful UDF for this stage."
    )
    fn_constructor_kwargs: Dict[str, Any] = Field(
        description="The keyword arguments of the UDF constructor."
    )
    map_batches_kwargs: Dict[str, Any] = Field(
        description="The arguments of .map_batches()."
    )

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
