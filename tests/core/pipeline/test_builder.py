"""
Tests for the PipelineBuilder class.

This module contains tests for the fluent interface for pipeline construction,
including stage addition, grouping, branching, and conditional execution.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from core.pipeline.builder import (
    PipelineBuilder,
    StageGroup,
    PipelineBranch,
    StageGroupError,
    BranchMergeError
)
from core.pipeline.pipeline import Pipeline
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext
from core.pipeline.factory import PipelineFactory


# Helper classes for testing
class TestStage(PipelineStage):
    """A test pipeline stage for testing."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Process the stage."""
        context.set('test_stage_output', 'processed')
        return True


# Fixtures
@pytest.fixture
def factory():
    """Create a mock PipelineFactory for testing."""
    return MagicMock(spec=PipelineFactory)


@pytest.fixture
def builder(factory):
    """Create a PipelineBuilder instance for testing."""
    return PipelineBuilder(factory)


# Tests for basic builder functionality
def test_builder_initialization(factory):
    """Test that the builder initializes properly."""
    builder = PipelineBuilder(factory)
    
    # Verify initial state
    assert builder.name == "pipeline"
    assert isinstance(builder.config, dict)
    assert len(builder.stages) == 0
    assert len(builder.groups) == 0
    assert builder.current_group is None
    assert len(builder.branches) == 0
    assert builder.active_branch is None
    assert builder.factory == factory


def test_set_name(builder):
    """Test setting the pipeline name."""
    result = builder.set_name("test_pipeline")
    
    # Verify the name was set
    assert builder.name == "test_pipeline"
    
    # Verify method chaining
    assert result == builder


def test_set_description(builder):
    """Test setting the pipeline description."""
    result = builder.set_description("Test pipeline description")
    
    # Verify the description was set
    assert builder.metadata["description"] == "Test pipeline description"
    
    # Verify method chaining
    assert result == builder


def test_set_version(builder):
    """Test setting the pipeline version."""
    result = builder.set_version("2.0.0")
    
    # Verify the version was set
    assert builder.metadata["version"] == "2.0.0"
    
    # Verify method chaining
    assert result == builder


def test_add_tags(builder):
    """Test adding tags to the pipeline."""
    result = builder.add_tags("tag1", "tag2")
    
    # Verify the tags were added
    assert "tag1" in builder.metadata["tags"]
    assert "tag2" in builder.metadata["tags"]
    
    # Verify method chaining
    assert result == builder
    
    # Test adding duplicate tags
    builder.add_tags("tag1", "tag3")
    assert builder.metadata["tags"].count("tag1") == 1
    assert "tag3" in builder.metadata["tags"]


def test_configure(builder):
    """Test configuring the pipeline."""
    result = builder.configure(
        parallel_execution=True,
        max_workers=10,
        continue_on_error=True
    )
    
    # Verify the configuration was set
    assert builder.config["parallel_execution"] is True
    assert builder.config["max_workers"] == 10
    assert builder.config["continue_on_error"] is True
    
    # Verify method chaining
    assert result == builder


def test_enable_parallel_execution(builder):
    """Test enabling parallel execution."""
    result = builder.enable_parallel_execution(max_workers=8)
    
    # Verify parallel execution was enabled
    assert builder.config["parallel_execution"] is True
    assert builder.config["max_workers"] == 8
    
    # Verify method chaining
    assert result == builder


def test_disable_parallel_execution(builder):
    """Test disabling parallel execution."""
    # First enable it
    builder.enable_parallel_execution()
    
    # Then disable it
    result = builder.disable_parallel_execution()
    
    # Verify parallel execution was disabled
    assert builder.config["parallel_execution"] is False
    
    # Verify method chaining
    assert result == builder


def test_continue_on_error(builder):
    """Test configuring continue on error."""
    result = builder.continue_on_error(True)
    
    # Verify the setting was applied
    assert builder.config["continue_on_error"] is True
    
    # Verify method chaining
    assert result == builder
    
    # Test disabling it
    builder.continue_on_error(False)
    assert builder.config["continue_on_error"] is False


def test_enable_monitoring(builder):
    """Test enabling monitoring."""
    result = builder.enable_monitoring(True)
    
    # Verify monitoring was enabled
    assert builder.config["enable_monitoring"] is True
    
    # Verify method chaining
    assert result == builder
    
    # Test disabling it
    builder.enable_monitoring(False)
    assert builder.config["enable_monitoring"] is False


# Tests for stage management
def test_add_stage(builder):
    """Test adding a stage to the pipeline."""
    stage = TestStage({"name": "test_stage"})
    result = builder.add_stage(stage)
    
    # Verify the stage was added
    assert len(builder.stages) == 1
    assert builder.stages[0] == stage
    
    # Verify method chaining
    assert result == builder


def test_add_stages(builder):
    """Test adding multiple stages to the pipeline."""
    stages = [
        TestStage({"name": "stage1"}),
        TestStage({"name": "stage2"}),
        TestStage({"name": "stage3"})
    ]
    
    result = builder.add_stages(stages)
    
    # Verify the stages were added
    assert len(builder.stages) == 3
    assert builder.stages == stages
    
    # Verify method chaining
    assert result == builder


# Tests for stage grouping
def test_begin_group(builder):
    """Test beginning a stage group."""
    result = builder.begin_group("test_group", "Test group description")
    
    # Verify the group was created
    assert builder.current_group is not None
    assert builder.current_group.name == "test_group"
    assert builder.current_group.description == "Test group description"
    assert "test_group" in builder.groups
    
    # Verify method chaining
    assert result == builder


def test_begin_group_already_active(builder):
    """Test that an error is raised when beginning a group when one is already active."""
    builder.begin_group("group1")
    
    # Verify the error is raised
    with pytest.raises(StageGroupError, match="Cannot begin a new group while another is active"):
        builder.begin_group("group2")


def test_end_group(builder):
    """Test ending a stage group."""
    builder.begin_group("test_group")
    result = builder.end_group()
    
    # Verify the group was ended
    assert builder.current_group is None
    
    # Verify method chaining
    assert result == builder


def test_end_group_no_active(builder):
    """Test that an error is raised when ending a group when none is active."""
    with pytest.raises(StageGroupError, match="No active group to end"):
        builder.end_group()


def test_add_stage_to_group(builder):
    """Test adding a stage to a group."""
    builder.begin_group("test_group")
    
    stage = TestStage({"name": "grouped_stage"})
    builder.add_stage(stage)
    
    # Verify the stage was added to the group
    assert len(builder.stages) == 0
    assert len(builder.current_group.stages) == 1
    assert builder.current_group.stages[0] == stage
    
    # End the group
    builder.end_group()


def test_add_group(builder):
    """Test adding a pre-defined group of stages."""
    stages = [
        TestStage({"name": "group_stage1"}),
        TestStage({"name": "group_stage2"})
    ]
    
    result = builder.add_group("test_group", stages, "Test group")
    
    # Verify the group was added
    assert "test_group" in builder.groups
    assert builder.groups["test_group"].name == "test_group"
    assert builder.groups["test_group"].description == "Test group"
    assert len(builder.groups["test_group"].stages) == 2
    
    # Verify method chaining
    assert result == builder


def test_include_group(builder):
    """Test including a named group in the pipeline."""
    # First create a group
    stages = [
        TestStage({"name": "include_stage1"}),
        TestStage({"name": "include_stage2"})
    ]
    builder.add_group("include_group", stages)
    
    # Then include it
    result = builder.include_group("include_group")
    
    # Verify the stages were added to the pipeline
    assert len(builder.stages) == 2
    assert builder.stages[0].config["name"] == "include_stage1"
    assert builder.stages[1].config["name"] == "include_stage2"
    
    # Verify method chaining
    assert result == builder


def test_include_nonexistent_group(builder):
    """Test that an error is raised when including a non-existent group."""
    with pytest.raises(StageGroupError, match="Group 'non_existent' does not exist"):
        builder.include_group("non_existent")


# Tests for branching
def test_begin_branch(builder):
    """Test beginning a branch."""
    result = builder.begin_branch("test_branch")
    
    # Verify the branch was created
    assert builder.active_branch == "test_branch"
    assert "test_branch" in builder.branches
    
    # Verify method chaining
    assert result == builder


def test_begin_branch_with_condition(builder):
    """Test beginning a branch with a condition."""
    # Create a condition function
    def condition(context):
        return context.get("flag", False)
    
    result = builder.begin_branch("conditional_branch", condition)
    
    # Verify the branch was created with the condition
    assert builder.active_branch == "conditional_branch"
    assert "conditional_branch" in builder.branches
    assert builder.branches["conditional_branch"].condition == condition
    
    # Verify method chaining
    assert result == builder


def test_begin_branch_already_active(builder):
    """Test that an error is raised when beginning a branch when one is already active."""
    builder.begin_branch("branch1")
    
    # Verify the error is raised
    with pytest.raises(BranchMergeError, match="Cannot begin a new branch while another is active"):
        builder.begin_branch("branch2")


def test_end_branch(builder):
    """Test ending a branch."""
    builder.begin_branch("test_branch")
    result = builder.end_branch()
    
    # Verify the branch was ended
    assert builder.active_branch is None
    
    # Verify method chaining
    assert result == builder


def test_end_branch_no_active(builder):
    """Test that an error is raised when ending a branch when none is active."""
    with pytest.raises(BranchMergeError, match="No active branch to end"):
        builder.end_branch()


def test_add_stage_to_branch(builder):
    """Test adding a stage to a branch."""
    builder.begin_branch("test_branch")
    
    stage = TestStage({"name": "branch_stage"})
    builder.add_stage(stage)
    
    # Verify the stage was added to the branch
    assert len(builder.stages) == 0
    assert len(builder.branches["test_branch"].stages) == 1
    assert builder.branches["test_branch"].stages[0] == stage
    
    # End the branch
    builder.end_branch()


def test_merge_branches(builder):
    """Test merging branches into the main pipeline."""
    # Create two branches with stages
    builder.begin_branch("branch1")
    builder.add_stage(TestStage({"name": "branch1_stage"}))
    builder.end_branch()
    
    builder.begin_branch("branch2")
    builder.add_stage(TestStage({"name": "branch2_stage"}))
    builder.end_branch()
    
    # Merge the branches
    result = builder.merge_branches("branch1", "branch2")
    
    # Verify the stages were added to the pipeline
    assert len(builder.stages) == 2
    assert builder.stages[0].config["name"] == "branch1_stage"
    assert builder.stages[1].config["name"] == "branch2_stage"
    
    # Verify method chaining
    assert result == builder


def test_merge_nonexistent_branch(builder):
    """Test that an error is raised when merging a non-existent branch."""
    with pytest.raises(BranchMergeError, match="Branch 'non_existent' does not exist"):
        builder.merge_branches("non_existent")


def test_conditional_stage(builder):
    """Test adding a conditional stage."""
    # Create a condition function
    def condition(context):
        return context.get("flag", False)
    
    stage = TestStage({"name": "conditional_stage"})
    result = builder.add_conditional_stage(stage, condition)
    
    # Verify a branch was created and merged
    assert len(builder.stages) == 1
    assert builder.stages[0] == stage
    
    # Verify method chaining
    assert result == builder


# Tests for pipeline creation
def test_create_pipeline(builder):
    """Test creating a pipeline from the builder."""
    # Configure the pipeline
    builder.set_name("test_pipeline")
    builder.add_stage(TestStage({"name": "stage1"}))
    builder.add_stage(TestStage({"name": "stage2"}))
    
    # Create the pipeline
    pipeline = builder.create_pipeline()
    
    # Verify the pipeline was created correctly
    assert pipeline.name == "test_pipeline"
    assert len(pipeline.stages) == 2
    assert pipeline.stages[0].config["name"] == "stage1"
    assert pipeline.stages[1].config["name"] == "stage2"


def test_build(builder):
    """Test building a pipeline configuration."""
    # Configure the pipeline
    builder.set_name("test_pipeline")
    builder.set_description("Test pipeline")
    builder.set_version("1.0.0")
    builder.add_tags("tag1", "tag2")
    builder.configure(parallel_execution=True)
    builder.add_stage(TestStage({"name": "stage1"}))
    builder.add_stage(TestStage({"name": "stage2"}))
    
    # Build the configuration
    config = builder.build()
    
    # Verify the configuration was built correctly
    assert config["name"] == "test_pipeline"
    assert config["description"] == "Test pipeline"
    assert config["version"] == "1.0.0"
    assert "tag1" in config["tags"]
    assert "tag2" in config["tags"]
    assert config["pipeline_config"]["parallel_execution"] is True
    assert len(config["stages"]) == 2
    assert config["stages"][0]["stage"] == "stage1"
    assert config["stages"][1]["stage"] == "stage2"


@pytest.mark.asyncio
async def test_build_and_create(factory):
    """Test building and creating a pipeline in one step."""
    # Create a mock factory
    pipeline = Pipeline("created_pipeline")
    factory.create_pipeline_from_config = AsyncMock(return_value=pipeline)
    
    # Create a builder with the mock factory
    builder = PipelineBuilder(factory)
    
    # Configure the pipeline
    builder.set_name("test_pipeline")
    builder.add_stage(TestStage({"name": "stage1"}))
    
    # Build and create the pipeline
    result = await builder.build_and_create()
    
    # Verify the factory was called
    factory.create_pipeline_from_config.assert_called_once()
    
    # Verify the result is the created pipeline
    assert result == pipeline


@pytest.mark.asyncio
async def test_build_and_create_no_factory(builder):
    """Test that an error is raised when trying to create without a factory."""
    # Remove the factory reference
    builder.factory = None
    
    # Verify the error is raised
    with pytest.raises(Exception, match="no factory provided"):
        await builder.build_and_create()


# Tests for StageGroup class
def test_stage_group():
    """Test the StageGroup class functionality."""
    group = StageGroup("test_group", "Test group description")
    
    # Test initial state
    assert group.name == "test_group"
    assert group.description == "Test group description"
    assert len(group.stages) == 0
    assert group.is_empty() is True
    
    # Test adding a stage
    stage = TestStage({"name": "group_stage"})
    group.add_stage(stage)
    assert len(group.stages) == 1
    assert group.stages[0] == stage
    assert group.is_empty() is False
    
    # Test adding multiple stages
    stages = [
        TestStage({"name": "group_stage2"}),
        TestStage({"name": "group_stage3"})
    ]
    group.add_stages(stages)
    assert len(group.stages) == 3
    
    # Test clearing stages
    group.clear()
    assert len(group.stages) == 0
    assert group.is_empty() is True


# Tests for PipelineBranch class
def test_pipeline_branch():
    """Test the PipelineBranch class functionality."""
    # Create a condition function
    def condition(context):
        return context.get("flag", False)
    
    branch = PipelineBranch("test_branch", condition)
    
    # Test initial state
    assert branch.name == "test_branch"
    assert branch.condition == condition
    assert len(branch.stages) == 0
    
    # Test adding a stage
    stage = TestStage({"name": "branch_stage"})
    branch.add_stage(stage)
    assert len(branch.stages) == 1
    assert branch.stages[0] == stage
    
    # Test adding multiple stages
    stages = [
        TestStage({"name": "branch_stage2"}),
        TestStage({"name": "branch_stage3"})
    ]
    branch.add_stages(stages)
    assert len(branch.stages) == 3
    
    # Test condition evaluation
    context_false = PipelineContext({"flag": False})
    assert branch.should_execute(context_false) is False
    
    context_true = PipelineContext({"flag": True})
    assert branch.should_execute(context_true) is True
    
    # Test branch without condition
    branch_no_condition = PipelineBranch("unconditional")
    assert branch_no_condition.should_execute(context_false) is True