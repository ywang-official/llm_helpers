#!/usr/bin/env python3
"""
PromptHelper - Advanced prompt management with automatic JSON formatting
"""

import os
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from ruamel.yaml import YAML
import logging
from jinja2 import Template, StrictUndefined, Environment, meta

logger = logging.getLogger(__name__)
yaml = YAML()


class ResponseType(Enum):
    """Enum for different LLM response types."""
    RAW_STRING = "raw_string"  # Return raw string response
    PARSED_JSON = "parsed_json"  # Return structured JSON response


class PromptHelper:
    """
    Helper class for loading and managing prompts with automatic JSON formatting.
    
    Supports loading prompts from YAML files or directories and automatically
    adds JSON response formatting based on the specified response type.
    """
    
    def __init__(self, 
                 prompt_source: Optional[Union[str, Path]] = None,
                 default_response_type: ResponseType = ResponseType.RAW_STRING):
        """
        Initialize the PromptHelper.
        
        Args:
            prompt_source: Path to prompt file/directory. If None, uses default prompts.
            default_response_type: Default response type for prompts
        """
        self.default_response_type = default_response_type
        self.prompts: Dict[str, Dict] = {}

        self.prompt_sources = [Path(__file__).parent.parent / "prompts" / "internal_prompts.yaml"]
        
        if prompt_source is None:
            # Use default prompts from the package
            prompt_source = Path(__file__).parent.parent / "prompts" / "sample_prompts.yaml"
        
        self.prompt_sources.append(Path(prompt_source))
        self.load_prompts()
    
    def load_prompts(self) -> None:
        """Load prompts from the specified source."""
        for prompt_source in self.prompt_sources:
            if prompt_source.is_file():
                self._load_from_file(prompt_source)
            elif prompt_source.is_dir():
                self._load_from_directory(prompt_source)
            else:
                raise ValueError(f"Prompt source not found: {prompt_source}")
        
        logger.info(f"Loaded {len(self.prompts)} prompt configurations")
    
    def _load_from_file(self, file_path: Path) -> None:
        """Load prompts from a single YAML file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.load(f)
                
            if isinstance(data, dict):
                self.prompts.update(data)
            else:
                raise ValueError(f"Invalid YAML structure in {file_path}")
                
        except Exception as e:
            logger.error(f"Error loading prompts from {file_path}: {e}")
            raise
    
    def _load_from_directory(self, dir_path: Path) -> None:
        """Load prompts from all YAML files in a directory."""
        yaml_files = list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml"))
        
        if not yaml_files:
            raise ValueError(f"No YAML files found in directory: {dir_path}")
        
        for yaml_file in yaml_files:
            self._load_from_file(yaml_file)
    
    def get_prompt_keys(self) -> List[str]:
        """Get all available prompt keys."""
        return list(self.prompts.keys())
    
    def get_prompt_config(self, prompt_key: str) -> Dict:
        """
        Get the raw prompt configuration for a key.
        
        Args:
            prompt_key: The prompt identifier
            
        Returns:
            Dict containing system_prompt, selected_parts_prompt, and config
        """
        if prompt_key not in self.prompts:
            raise ValueError(f"Prompt key '{prompt_key}' not found. Available: {self.get_prompt_keys()}")
        
        return self.prompts[prompt_key].copy()
    
    def build_system_prompt(self, 
                           prompt_key: str,
                           response_type: Optional[ResponseType] = None,
                           custom_schema: Optional[Dict] = None,
                           **kwargs) -> str:
        """
        Build the system prompt with automatic JSON formatting instructions.
        
        Args:
            prompt_key: The prompt identifier
            response_type: Type of response expected (overrides default)
            custom_schema: Custom JSON schema for structured responses
            
        Returns:
            Complete system prompt with JSON formatting instructions
        """
        prompt_config = self.get_prompt_config(prompt_key)
        system_prompt = prompt_config.get("system_prompt", "")
        
        # Use config-level response_type/custom_schema if not overridden
        response_type = self._get_response_type(response_type or self._get_config_response_type(prompt_config) or self.default_response_type)
        custom_schema = custom_schema or prompt_config.get("custom_schema")
        
        # Optionally render system prompt with Jinja2
        if kwargs:
            try:
                system_prompt = Template(system_prompt).render(**kwargs)
            except Exception as e:
                logger.warning(f"Jinja2 render error in system_prompt: {e}")
        
        # Add JSON formatting instructions
        json_instructions = self._get_json_instructions(prompt_key, response_type, custom_schema)
        
        if json_instructions:
            system_prompt += "\n\n" + json_instructions
        
        # Add formatted examples if present
        examples = prompt_config.get("examples")
        if examples:
            system_prompt += "\n\n" + self._format_examples(examples, prompt_key, response_type)
        
        return system_prompt
    
    def build_user_prompt(self, prompt_key: str, **kwargs) -> str:
        """
        Build the user prompt with variable substitution.
        
        Args:
            prompt_key: The prompt identifier
            **kwargs: Variables to substitute in the prompt template
            
        Returns:
            User prompt with variables substituted
        """
        prompt_config = self.get_prompt_config(prompt_key)
        user_prompt = prompt_config.get("selected_parts_prompt", "")
        
        try:
            user_prompt = Template(user_prompt).render(**kwargs)
        except Exception as e:
            logger.warning(f"Jinja2 render error in user_prompt: {e}")
        
        return user_prompt
    
    def _get_json_instructions(self, 
                              prompt_key: str,
                              response_type: ResponseType,
                              custom_schema: Optional[Dict] = None) -> str:
        """Generate JSON formatting instructions based on response type."""
        
        if response_type == ResponseType.RAW_STRING:
            response_format_string = '"Your response content as a string"'
        elif response_type == ResponseType.PARSED_JSON:
            if custom_schema:
                response_format_string = self._format_schema_example(custom_schema)
            else:
                response_format_string = '{ //Your response content as a json object here \n}'
        
        return f"""
Your task: return a valid JSON object with the format below. All valid json content should be wrapped with __json_start__ and __json_end__.
- JSON object with key "{prompt_key}":
{{
  "{prompt_key}": {response_format_string}
}}

<expected response>
__json_start__
{{
  "{prompt_key}": <your response content here>
}}
__json_end__
</expected response>"""
    
    def _format_schema_example(self, schema: Dict) -> str:
        """Format a JSON schema as an example."""
        def format_value(value):
            if isinstance(value, dict):
                return "{" + ", ".join(f'"{k}": {format_value(v)}' for k, v in value.items()) + "}"
            elif isinstance(value, list):
                return f"[{format_value(value[0]) if value else '...'}]"
            elif isinstance(value, str):
                return f'"{value}"'
            else:
                return str(value)
        
        return format_value(schema)
    
    def _format_examples(self, examples: List[Dict[str, Any]], prompt_key: str, response_type: ResponseType) -> str:
        """Format examples as a readable string for the system prompt."""
        formatted = ["Examples:"]
        for i, ex in enumerate(examples, 1):
            input_str = ex["input"].strip()
            output_str = ex["output"].strip()
            formatted.append(f"<example_{i}_input>\n{input_str}\n</example_{i}_input>")
            if response_type == ResponseType.PARSED_JSON:
                formatted.append(f"<example_{i}_output>\n__json_start__\n{{ \"{prompt_key}\": {output_str} }}\n__json_end__\n</example_{i}_output>")
            elif response_type == ResponseType.RAW_STRING:
                formatted.append(f"<example_{i}_output>\n__json_start__\n{{ \"{prompt_key}\": \"{output_str}\" }}\n__json_end__\n</example_{i}_output>\n")
        return "\n".join(formatted)

    def _get_response_type(self, val: Union[str, ResponseType]) -> Optional[ResponseType]:
        if val is None:
            return None
        if isinstance(val, ResponseType):
            return val
        try:
            return ResponseType(val)
        except Exception:
            return None

    def _get_config_response_type(self, prompt_config: Dict) -> Optional[ResponseType]:
        val = prompt_config.get("response_type")
        return self._get_response_type(val)

    def get_prompt_bundle(self, 
                         prompt_key: str,
                         response_type: Optional[ResponseType] = None,
                         custom_schema: Optional[Dict] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        Get a complete prompt bundle ready for LLM processing.
        
        Args:
            prompt_key: The prompt identifier
            response_type: Type of response expected. This can be provided as an argument, or in the prompt config itself.
              If provided as an argument, it will override the default response type.
            custom_schema: Custom JSON schema for structured responses. This can be provided as an argument, or in the prompt config itself.
              If provided as an argument, it will override the default schema.
            **kwargs: Variables to substitute in prompts
            
        Returns:
            Dict containing system_prompt, user_prompt, config, and metadata
        """
        prompt_config = self.get_prompt_config(prompt_key)
        # Use config-level response_type/custom_schema if not overridden
        response_type = self._get_response_type(response_type or self._get_config_response_type(prompt_config) or self.default_response_type)
        custom_schema = custom_schema or prompt_config.get("custom_schema")

        return {
            "system_prompt": self.build_system_prompt(prompt_key, response_type, custom_schema, **kwargs),
            "user_prompt": self.build_user_prompt(prompt_key, **kwargs),
            "config": prompt_config.get("config", {}),
            "metadata": {
                "prompt_key": prompt_key,
                "response_type": response_type,
                "custom_schema": custom_schema
            }
        }
    
    def add_custom_prompt(self, 
                         prompt_key: str,
                         system_prompt: str,
                         user_prompt: str,
                         config: Optional[Dict] = None,
                         response_type: Optional[ResponseType] = None,
                         custom_schema: Optional[Dict] = None,
                         examples: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add a custom prompt at runtime.
        
        Args:
            prompt_key: Unique identifier for the prompt
            system_prompt: System prompt template
            user_prompt: User prompt template
            config: Optional configuration dict
        """
        self.prompts[prompt_key] = {
            "system_prompt": system_prompt,
            "selected_parts_prompt": user_prompt,
            "config": config or {},
            "response_type": response_type.value if isinstance(response_type, ResponseType) else response_type,
            "custom_schema": custom_schema,
            "examples": examples or []
        }
        
        logger.info(f"Added custom prompt: {prompt_key}")
    
    def list_prompts(self) -> Dict[str, Dict]:
        """List all available prompts with their configurations."""
        return {
            key: {
                "has_system_prompt": bool(config.get("system_prompt")),
                "has_user_prompt": bool(config.get("selected_parts_prompt")),
                "config_keys": list(config.get("config", {}).keys()) if config.get("config") else [],
                "has_examples": bool(config.get("examples")),
                "response_type": config.get("response_type"),
                "custom_schema": bool(config.get("custom_schema")),
            }
            for key, config in self.prompts.items()
        }