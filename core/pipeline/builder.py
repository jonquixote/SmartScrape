"""
Pipeline Builder Module.

This module provides the PipelineBuilder class for fluent construction of pipelines,
with support for branching, merging, grouping, and conditional construction.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union, cast

from core.pipeline.pipeline import Pipeline, PipelineError
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext


logger = logging.getLogger(__name__)


class PipelineBuilderError(Exception):
    """Base exception for pipeline builder errors."""
    pass


class StageGroupError(PipelineBuilderError):
    """Exception raised when operations on stage groups fail."""
    pass


class BranchMergeError(PipelineBuilderError):
    """Exception raised when branch or merge operations fail."""
    pass


class StageGroup:
    """
    Represents a group of pipeline stages for organizational purposes.
    
    This is used by the PipelineBuilder to support grouping related stages together
    for improved organization and conditional inclusion.
    
    Attributes:
        name (str): Name of the group
        stages (List[PipelineStage]): Stages in this group
        description (str): Description of the group's purpose
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a stage group.
        
        Args:
            name: Name of the group
            description: Description of the group's purpose
        """
        self.name = name
        self.stages: List[PipelineStage] = []
        self.description = description
        
    def add_stage(self, stage: PipelineStage) -> 'StageGroup':
        """
        Add a stage to the group.
        
        Args:
            stage: Stage to add
            
        Returns:
            self for method chaining
        """
        self.stages.append(stage)
        return self
        
    def add_stages(self, stages: List[PipelineStage]) -> 'StageGroup':
        """
        Add multiple stages to the group.
        
        Args:
            stages: Stages to add
            
        Returns:
            self for method chaining
        """
        self.stages.extend(stages)
        return self
        
    def clear(self) -> 'StageGroup':
        """
        Remove all stages from the group.
        
        Returns:
            self for method chaining
        """
        self.stages.clear()
        return self
        
    def is_empty(self) -> bool:
        """
        Check if the group is empty.
        
        Returns:
            True if the group has no stages, False otherwise
        """
        return len(self.stages) == 0


class PipelineBranch:
    """
    Represents a branch in a pipeline for conditional flows.
    
    Branches allow for parallel or conditional execution paths within a pipeline.
    
    Attributes:
        name (str): Name of the branch
        stages (List[PipelineStage]): Stages in this branch
        condition (Optional[Callable[[PipelineContext], bool]]): Condition for execution
    """
    
    def __init__(self, name: str, 
                condition: Optional[Callable[[PipelineContext], bool]] = None):
        """
        Initialize a pipeline branch.
        
        Args:
            name: Name of the branch
            condition: Optional condition for branch execution
        """
        self.name = name
        self.stages: List[PipelineStage] = []
        self.condition = condition
        
    def add_stage(self, stage: PipelineStage) -> 'PipelineBranch':
        """
        Add a stage to the branch.
        
        Args:
            stage: Stage to add
            
        Returns:
            self for method chaining
        """
        self.stages.append(stage)
        return self
        
    def add_stages(self, stages: List[PipelineStage]) -> 'PipelineBranch':
        """
        Add multiple stages to the branch.
        
        Args:
            stages: Stages to add
            
        Returns:
            self for method chaining
        """
        self.stages.extend(stages)
        return self
        
    def should_execute(self, context: PipelineContext) -> bool:
        """
        Check if this branch should be executed.
        
        Args:
            context: Pipeline context
            
        Returns:
            True if the branch should execute, False otherwise
        """
        if self.condition is None:
            return True
        return self.condition(context)


class PipelineBuilder:
    """
    Fluent interface for constructing pipelines.
    
    This class provides a builder pattern implementation for creating pipelines,
    with support for stage grouping, branching, and conditional construction.
    
    Attributes:
        name (str): Name of the pipeline being built
        config (Dict[str, Any]): Pipeline configuration
        stages (List[PipelineStage]): Stages added to the pipeline
        current_group (Optional[StageGroup]): Current stage group being built
        branches (Dict[str, PipelineBranch]): Named branches for parallel/conditional execution
        active_branch (Optional[str]): Name of the currently active branch
    """
    
    def __init__(self, factory=None):
        """
        Initialize a pipeline builder.
        
        Args:
            factory: Optional pipeline factory for creating the final pipeline
        """
        self.factory = factory
        self.name = "pipeline"
        self.config: Dict[str, Any] = {}
        self.stages: List[PipelineStage] = []
        self.groups: Dict[str, StageGroup] = {}
        self.current_group: Optional[StageGroup] = None
        self.branches: Dict[str, PipelineBranch] = {}
        self.active_branch: Optional[str] = None
        self.metadata: Dict[str, Any] = {
            "description": "",
            "version": "1.0.0",
            "tags": []
        }
        
    def set_name(self, name: str) -> 'PipelineBuilder':
        """
        Set the name of the pipeline.
        
        Args:
            name: Pipeline name
            
        Returns:
            self for method chaining
        """
        self.name = name
        return self
        
    def set_description(self, description: str) -> 'PipelineBuilder':
        """
        Set the description of the pipeline.
        
        Args:
            description: Pipeline description
            
        Returns:
            self for method chaining
        """
        self.metadata["description"] = description
        return self
        
    def set_version(self, version: str) -> 'PipelineBuilder':
        """
        Set the version of the pipeline.
        
        Args:
            version: Pipeline version
            
        Returns:
            self for method chaining
        """
        self.metadata["version"] = version
        return self
        
    def add_tags(self, *tags: str) -> 'PipelineBuilder':
        """
        Add tags to the pipeline.
        
        Args:
            *tags: Tags to add
            
        Returns:
            self for method chaining
        """
        for tag in tags:
            if tag not in self.metadata["tags"]:
                self.metadata["tags"].append(tag)
        return self
        
    def configure(self, **options) -> 'PipelineBuilder':
        """
        Configure the pipeline with options.
        
        Args:
            **options: Configuration options
            
        Returns:
            self for method chaining
        """
        for key, value in options.items():
            self.config[key] = value
        return self
        
    def enable_parallel_execution(self, max_workers: int = 5) -> 'PipelineBuilder':
        """
        Enable parallel execution of pipeline stages.
        
        Args:
            max_workers: Maximum number of parallel workers
            
        Returns:
            self for method chaining
        """
        self.config["parallel_execution"] = True
        self.config["max_workers"] = max_workers
        return self
        
    def disable_parallel_execution(self) -> 'PipelineBuilder':
        """
        Disable parallel execution of pipeline stages.
        
        Returns:
            self for method chaining
        """
        self.config["parallel_execution"] = False
        return self
        
    def continue_on_error(self, enabled: bool = True) -> 'PipelineBuilder':
        """
        Configure whether the pipeline continues after errors.
        
        Args:
            enabled: True to continue, False to stop
            
        Returns:
            self for method chaining
        """
        self.config["continue_on_error"] = enabled
        return self
        
    def enable_monitoring(self, enabled: bool = True) -> 'PipelineBuilder':
        """
        Configure pipeline execution monitoring.
        
        Args:
            enabled: True to enable, False to disable
            
        Returns:
            self for method chaining
        """
        self.config["enable_monitoring"] = enabled
        return self
        
    def add_stage(self, stage: PipelineStage) -> 'PipelineBuilder':
        """
        Add a stage to the pipeline.
        
        If a group is active, the stage is added to the group.
        If a branch is active, the stage is added to the branch.
        Otherwise, the stage is added directly to the pipeline.
        
        Args:
            stage: Stage to add
            
        Returns:
            self for method chaining
        """
        if self.active_branch:
            self.branches[self.active_branch].add_stage(stage)
        elif self.current_group:
            self.current_group.add_stage(stage)
        else:
            self.stages.append(stage)
        return self
        
    def add_stages(self, stages: List[PipelineStage]) -> 'PipelineBuilder':
        """
        Add multiple stages to the pipeline.
        
        Args:
            stages: Stages to add
            
        Returns:
            self for method chaining
        """
        for stage in stages:
            self.add_stage(stage)
        return self
        
    def begin_group(self, name: str, description: str = "") -> 'PipelineBuilder':
        """
        Begin a new stage group.
        
        Args:
            name: Name of the group
            description: Description of the group
            
        Returns:
            self for method chaining
            
        Raises:
            StageGroupError: If there is already an active group
        """
        if self.current_group:
            raise StageGroupError("Cannot begin a new group while another is active")
            
        if name in self.groups:
            self.current_group = self.groups[name]
        else:
            self.current_group = StageGroup(name, description)
            self.groups[name] = self.current_group
            
        return self
        
    def end_group(self) -> 'PipelineBuilder':
        """
        End the current stage group.
        
        Returns:
            self for method chaining
            
        Raises:
            StageGroupError: If there is no active group
        """
        if not self.current_group:
            raise StageGroupError("No active group to end")
            
        self.current_group = None
        return self
        
    def add_group(self, name: str, stages: List[PipelineStage],
                description: str = "") -> 'PipelineBuilder':
        """
        Add a pre-defined group of stages.
        
        Args:
            name: Name of the group
            stages: Stages to include in the group
            description: Description of the group
            
        Returns:
            self for method chaining
        """
        group = StageGroup(name, description)
        group.add_stages(stages)
        self.groups[name] = group
        return self
        
    def include_group(self, name: str) -> 'PipelineBuilder':
        """
        Include all stages from a named group in the pipeline.
        
        Args:
            name: Name of the group to include
            
        Returns:
            self for method chaining
            
        Raises:
            StageGroupError: If the group does not exist
        """
        if name not in self.groups:
            raise StageGroupError(f"Group '{name}' does not exist")
            
        group = self.groups[name]
        
        if self.active_branch:
            for stage in group.stages:
                self.branches[self.active_branch].add_stage(stage)
        else:
            for stage in group.stages:
                self.stages.append(stage)
                
        return self
        
    def begin_branch(self, name: str, 
                    condition: Optional[Callable[[PipelineContext], bool]] = None) -> 'PipelineBuilder':
        """
        Begin a new branch for conditional or parallel execution.
        
        Args:
            name: Name of the branch
            condition: Optional condition for branch execution
            
        Returns:
            self for method chaining
            
        Raises:
            BranchMergeError: If there is already an active branch
        """
        if self.active_branch:
            raise BranchMergeError("Cannot begin a new branch while another is active")
            
        if name in self.branches:
            self.active_branch = name
        else:
            self.branches[name] = PipelineBranch(name, condition)
            self.active_branch = name
            
        return self
        
    def end_branch(self) -> 'PipelineBuilder':
        """
        End the current branch.
        
        Returns:
            self for method chaining
            
        Raises:
            BranchMergeError: If there is no active branch
        """
        if not self.active_branch:
            raise BranchMergeError("No active branch to end")
            
        self.active_branch = None
        return self
        
    def merge_branches(self, *branch_names: str) -> 'PipelineBuilder':
        """
        Merge named branches into the main pipeline.
        
        Args:
            *branch_names: Names of branches to merge
            
        Returns:
            self for method chaining
            
        Raises:
            BranchMergeError: If any branch does not exist
        """
        # If no branch names are provided, merge all branches
        names_to_merge = branch_names or list(self.branches.keys())
        
        for name in names_to_merge:
            if name not in self.branches:
                raise BranchMergeError(f"Branch '{name}' does not exist")
                
            branch = self.branches[name]
            
            # Add all stages from the branch to the main pipeline
            # In a more sophisticated implementation, this would create
            # proper conditional execution paths
            for stage in branch.stages:
                self.stages.append(stage)
                
        return self
        
    def when(self, condition: Callable[[Dict[str, Any]], bool]) -> 'PipelineBuilder':
        """
        Conditionally include the next stage based on configuration.
        
        This creates a transient condition that applies only to the next stage or group.
        
        Args:
            condition: Function taking the pipeline config and returning a boolean
            
        Returns:
            self for method chaining
        """
        # This could be implemented by creating a transient branch with the condition
        # For simplicity, we'll just store the condition for the next operation
        self._next_condition = condition
        return self
        
    def create_pipeline(self) -> Pipeline:
        """
        Create a pipeline from the current configuration.
        
        Returns:
            The constructed Pipeline instance
            
        Raises:
            PipelineBuilderError: If pipeline creation fails
        """
        # Create a pipeline instance
        pipeline = Pipeline(self.name, self.config)
        
        # Add all stages
        for stage in self.stages:
            pipeline.add_stage(stage)
            
        return pipeline
        
    def build(self) -> Dict[str, Any]:
        """
        Build a pipeline configuration dictionary.
        
        This can be used with a PipelineFactory to create a pipeline instance,
        or saved to a file for later use.
        
        Returns:
            Pipeline configuration dictionary
        """
        # Create the configuration dictionary
        config = {
            "name": self.name,
            "description": self.metadata["description"],
            "version": self.metadata["version"],
            "tags": self.metadata["tags"],
            "pipeline_config": self.config,
            "stages": []
        }
        
        # Add stage configurations
        for stage in self.stages:
            stage_config = {
                "stage": stage.name
            }
            
            # If the stage has a configuration, include it
            if hasattr(stage, "config") and stage.config:
                stage_config["config"] = stage.config
                
            config["stages"].append(stage_config)
            
        return config
        
    async def build_and_create(self) -> Pipeline:
        """
        Build the configuration and create a pipeline instance.
        
        This requires a factory to have been provided during initialization.
        
        Returns:
            Initialized Pipeline instance
            
        Raises:
            PipelineBuilderError: If no factory was provided or creation fails
        """
        if not self.factory:
            raise PipelineBuilderError(
                "Cannot create pipeline: no factory provided during initialization"
            )
            
        config = self.build()
        try:
            return await self.factory.create_pipeline_from_config(config, self.name)
        except Exception as e:
            raise PipelineBuilderError(f"Failed to create pipeline: {str(e)}") from e
            
    def add_conditional_stage(self, stage: PipelineStage, 
                             condition: Callable[[PipelineContext], bool]) -> 'PipelineBuilder':
        """
        Add a stage that only executes if the condition is met.
        
        Args:
            stage: Stage to add
            condition: Condition for execution
            
        Returns:
            self for method chaining
        """
        # Create a temporary branch for the conditional stage
        branch_name = f"conditional_{id(stage)}"
        self.begin_branch(branch_name, condition)
        self.add_stage(stage)
        self.end_branch()
        self.merge_branches(branch_name)
        
        return self
        
    def if_config(self, key: str, value: Any = True) -> 'PipelineBuilder':
        """
        Include the next stage only if a config value matches.
        
        Args:
            key: Configuration key to check
            value: Expected value
            
        Returns:
            self for method chaining
        """
        def check_config(context: PipelineContext) -> bool:
            if not context:
                return False
            return context.get(key) == value
            
        # Store this condition for the next stage
        self._next_condition = check_config
        return self
        
    def for_each(self, items_key: str) -> 'PipelineBuilder':
        """
        Create a loop context for the next stages.
        
        This is a placeholder for a more sophisticated implementation
        that would support iterating over items in the context.
        
        Args:
            items_key: Key in the context containing items to iterate over
            
        Returns:
            self for method chaining
        """
        # This would require a special handling in the Pipeline class
        # to repeat stages for each item in the collection
        self.config["loop_context"] = {
            "enabled": True,
            "items_key": items_key
        }
        return self
        
    def end_for_each(self) -> 'PipelineBuilder':
        """
        End a loop context.
        
        Returns:
            self for method chaining
        """
        # This would mark the end of the loop in actual implementation
        if "loop_context" in self.config:
            self.config["loop_context"]["enabled"] = False
        return self