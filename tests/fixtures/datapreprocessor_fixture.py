"""Dataloader fixture."""

import pandas as pd
import pytest
from databricks.connect import DatabricksSession
from loguru import logger
from pyspark.sql import SparkSession

from house_price import PROJECT_DIR
from house_price.config import ProjectConfig, Tags


@pytest.fixture(scope="session")
def spark_session() -> SparkSession:
    """Create and return a remote Databricks Spark session for tests."""
    spark = DatabricksSession.builder.serverless().profile("dev").getOrCreate()

    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    """Load and return the project configuration.

    This fixture reads the project configuration from a YAML file and returns a ProjectConfig object.

    :return: The loaded project configuration
    """
    config_file_path = (PROJECT_DIR / "project_config.yml").resolve()
    logger.info(f"Current config file path: {config_file_path.as_posix()}")
    config = ProjectConfig.from_yaml(config_file_path.as_posix())
    return config


@pytest.fixture(scope="function")
def sample_data(config: ProjectConfig, spark_session: SparkSession) -> pd.DataFrame:
    """Create a sample DataFrame from a CSV file.

    This fixture reads a CSV file using either Spark or pandas, then converts it to a Pandas DataFrame,

    :return: A sampled Pandas DataFrame containing some sample of the original data.
    """
    file_path = PROJECT_DIR / "tests" / "test_data" / "sample.csv"
    sample = pd.read_csv(file_path.as_posix())

    # Alternative approach to reading the sample
    # Important Note: Replace NaN with None in Pandas Before Conversion to Spark DataFrame:
    # sample = sample.where(sample.notna(), None)  # noqa
    # sample = spark_session.createDataFrame(sample).toPandas()  # noqa
    return sample


@pytest.fixture(scope="session")
def tags() -> Tags:
    """Create and return a Tags instance for the test session.

    This fixture provides a Tags object with predefined values for git_sha, branch, and job_run_id.
    """
    return Tags(git_sha="wxyz", branch="test", job_run_id="9")
