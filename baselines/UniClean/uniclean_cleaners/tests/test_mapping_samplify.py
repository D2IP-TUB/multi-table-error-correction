from types import SimpleNamespace
import unittest

from pyspark.sql import SparkSession

from CoreSetSample.mapping_samplify import block_sample


class BlockSampleTest(unittest.TestCase):
    def test_block_sample_accepts_boolean_target_columns(self):
        spark = SparkSession.builder.master("local[1]").appName("uniclean-test").getOrCreate()
        try:
            data = spark.createDataFrame(
                [(0, "a", True), (1, "a", False)],
                ["index", "record_key", "uncertain"],
            )
            model = SimpleNamespace(
                source={"record_key"},
                target={"uncertain"},
                fixValueRules={},
            )

            sample, _ = block_sample(data, [model])

            self.assertEqual(sample.schema["uncertain"].dataType.typeName(), "string")
        finally:
            spark.stop()
