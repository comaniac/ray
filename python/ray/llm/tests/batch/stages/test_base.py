import pytest
from typing import Dict, Any, AsyncIterator
from ray.llm._internal.batch.stages.base import (
    wrap_preprocess,
    wrap_postprocess,
    StatefulStage,
    StatefulStageUDF,
)


def test_wrap_preprocess():
    # Test function that doubles a number
    def double(x: dict) -> int:
        return x["value"] * 2

    # Test with carry_over=False
    wrapped = wrap_preprocess(double, "input", carry_over=False)
    result = wrapped({"value": 5, "extra": "data"})
    assert result == {"input": 10}

    # Test with carry_over=True
    wrapped = wrap_preprocess(double, "input", carry_over=True)
    result = wrapped({"value": 5, "extra": "data"})
    assert result == {"input": 10, "extra": "data", "value": 5}


def test_wrap_postprocess():
    # Test function that converts number to string
    def to_string(x: int) -> dict:
        return {"result": str(x)}

    # Test with carry_over=False
    wrapped = wrap_postprocess(to_string, "input", carry_over=False)
    result = wrapped({"input": 42, "extra": "data"})
    assert result == {"result": "42"}

    # Test with carry_over=True
    wrapped = wrap_postprocess(to_string, "input", carry_over=True)
    result = wrapped({"input": 42, "extra": "data"})
    assert result == {"result": "42", "extra": "data"}

    # Test missing input column
    with pytest.raises(ValueError):
        wrapped({"wrong_key": 42})


class TestStatefulStageUDF:
    class SimpleUDF(StatefulStageUDF):
        async def udf(self, rows: list[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
            for row in rows:
                yield {"processed": row["value"] * 2}

        @property
        def expected_input_keys(self) -> Dict[str, StatefulStageUDF.InputKeyType]:
            return {"value": StatefulStageUDF.InputKeyType.REQUIRED}

    @pytest.mark.asyncio
    async def test_basic_processing(self):
        udf = self.SimpleUDF(
            input_column="input_col",
            output_column="output_col",
            carry_over=True,
        )

        batch = {
            "input_col": [{"value": 1}, {"value": 2}],
            "extra": ["a", "b"],
        }

        results = []
        async for result in udf(batch):
            results.append(result)

        assert len(results) == 2
        assert results[0] == {
            "output_col": [{"processed": 2}],
            "extra": ["a"],
        }
        assert results[1] == {
            "output_col": [{"processed": 4}],
            "extra": ["b"],
        }

    @pytest.mark.asyncio
    async def test_missing_input_column(self):
        udf = self.SimpleUDF(
            input_column="wrong_col",
            output_column="output_col",
            carry_over=True,
        )

        batch = {
            "input_col": [{"value": 1}],
            "extra": ["a"],
        }

        with pytest.raises(ValueError):
            async for _ in udf(batch):
                pass

    @pytest.mark.asyncio
    async def test_missing_required_key(self):
        udf = self.SimpleUDF(
            input_column="input_col",
            output_column="output_col",
            carry_over=True,
        )

        batch = {
            "input_col": [{"wrong_key": 1}],
            "extra": ["a"],
        }

        with pytest.raises(ValueError):
            async for _ in udf(batch):
                pass


def test_stateful_stage():
    udf = TestStatefulStageUDF.SimpleUDF(
        input_column="input",
        output_column="output",
        carry_over=True,
    )

    stage = StatefulStage(
        fn=udf,
        fn_constructor_kwargs={"input_column": "input"},
        map_batches_kwargs={"batch_size": 10},
    )

    assert stage.fn == udf
    assert stage.fn_constructor_kwargs == {"input_column": "input"}
    assert stage.map_batches_kwargs == {"batch_size": 10}
