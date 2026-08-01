import tempfile
import unittest
from pathlib import Path

from pyspark.sql import SparkSession

from util import save_cleaned_data


class SaveCleanedDataTest(unittest.TestCase):
    def test_save_cleaned_data_accepts_boolean_columns(self):
        spark = SparkSession.builder.master("local[1]").appName("uniclean-test").getOrCreate()
        try:
            data = spark.createDataFrame([(0, True)], ["index", "uncertain"])

            with tempfile.TemporaryDirectory() as tmpdir:
                save_cleaned_data(data, tmpdir, "table")

                self.assertTrue(Path(tmpdir, "tableCleaned.csv").is_file())
        finally:
            spark.stop()
