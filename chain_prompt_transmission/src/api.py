from typing import Dict, Optional, Any, Union, List  ### CHANGES: added List
import logging
from dataclasses import dataclass
import json
from pathlib import Path
import yaml

# If this import is truly your Deepseek wrapper, keep it.
# Otherwise, rename as needed:
from openai import OpenAI, OpenAIError

@dataclass
class APIConfig:
    """Configuration class for API settings."""
    model: str
    max_tokens: int
    temperature: float
    base_url: str

class APIClient:
    """Handles all API interactions with the language model (Deepseek)."""
    
    def __init__(self, api_key: str, config_path: Path):
        """
        Initialize the API client.
        
        Args:
            api_key (str): The API key for authentication
            config_path (Path): Path to the configuration file
            
        Raises:
            ValueError: If API key is missing or invalid
            FileNotFoundError: If config file not found
        """
        if not api_key:
            raise ValueError("API key is required")
        
        # Load YAML config
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        # Basic validation of 'api' section
        if 'api' not in raw_config:
            raise ValueError("Missing 'api' section in configuration file.")
        
        self.config = APIConfig(**raw_config['api'])

        # Initialize the client with the base_url from config
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.base_url
        )
        
    def make_call(
        self,
        prompt: str,
        article_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make an API call to the language model (Deepseek).
        
        Args:
            prompt (str): The prompt to send
            article_id (Optional[str]): Article identifier for logging
            
        Returns:
            Dict[str, Any]: Parsed JSON response or error dictionary
        """
        try:
            logging.info(
                f"API request for article {article_id}",
                extra={"article_id": article_id, "prompt_length": len(prompt)}
            )
            
            # Equivalent to: client.chat.completions.create(...)
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an analytical assistant extracting verbatim statements."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=False
            )
            
            content = response.choices[0].message.content.strip()
            logging.info(
                f"Raw API response for article {article_id}:\n{content}",
                extra={"article_id": article_id, "response_length": len(content)}
            )
            
            # Clean response if triple-backtick fenced
            if content.startswith("```") and content.endswith("```"):
                content = "\n".join(content.split("\n")[1:-1]).strip()
            
            try:
                # Attempt JSON parse
                parsed_response = json.loads(content)
                
                # Ensure "individual_statements" key always exists
                if "individual_statements" not in parsed_response:
                    parsed_response["individual_statements"] = []
                
                # Convert any top-level affirm_/deny_ fields into "individual_statements"
                for key in list(parsed_response.keys()):
                    if key.startswith(("affirm_", "deny_")) and isinstance(parsed_response[key], str):
                        statement = parsed_response[key]
                        # Attempt to parse out "affirm_transmissible_statement_1", for example
                        stmt_type = key.split("_statement_")[0]  # e.g. "affirm_transmissible"
                        stmt_num = key.split("_")[-1]            # e.g. "1"
                        
                        parsed_response["individual_statements"].append({
                            "statement": statement,
                            "statement_id": f"{stmt_type}_statement_{stmt_num}",
                            "explanation": f"Statement {stmt_num} of type {stmt_type}"
                        })
                
                logging.info(
                    f"Successfully parsed JSON response for article {article_id}",
                    extra={"article_id": article_id}
                )
                return parsed_response
                
            except json.JSONDecodeError as e:
                logging.error(
                    f"JSON parse error for article {article_id}: {str(e)}\nContent: {content}",
                    extra={"article_id": article_id},
                    exc_info=True
                )
                return {"error": f"JSON parse error: {str(e)}", "individual_statements": []}
        
        except OpenAIError as e:
            # Handle any API-level errors from your Deepseek/OpenAI wrapper
            logging.error(
                f"API error for article {article_id}: {str(e)}",
                extra={
                    "article_id": article_id,
                    "error_type": type(e).__name__,
                    "error_details": str(e)
                },
                exc_info=True
            )
            return {"error": f"API error: {str(e)}", "individual_statements": []}
            
        except Exception as e:
            # Catch-all for unexpected exceptions
            logging.error(
                f"Unexpected error in API call for article {article_id}: {str(e)}",
                extra={"article_id": article_id, "error_type": type(e).__name__},
                exc_info=True
            )
            return {"error": f"Unexpected error: {str(e)}", "individual_statements": []}

    def execute_chain(
        self,
        prompts: Dict[str, str],
        article_id: str
    ) -> Dict[str, Any]:
        """
        Execute a chain of prompts in sequence.
        
        Args:
            prompts (Dict[str, str]): Dictionary of prompt_name → prompt_text
            article_id (str): Article identifier for logging
            
        Returns:
            Dict[str, Any]: Combined results from all prompts or error
        """
        results = {}
        try:
            for prompt_name, prompt in prompts.items():
                result = self.make_call(prompt, article_id)
                if "error" in result:
                    logging.error(
                        f"Chain execution failed at {prompt_name} for article {article_id}",
                        extra={"article_id": article_id, "prompt": prompt_name}
                    )
                    return {
                        "error": f"Chain failed at {prompt_name}",
                        "individual_statements": []
                    }
                results[prompt_name] = result
                
            return results
            
        except Exception as e:
            logging.error(
                f"Chain execution error for article {article_id}: {str(e)}",
                extra={"article_id": article_id},
                exc_info=True
            )
            return {"error": f"Chain execution error: {str(e)}", "individual_statements": []}

    def validate_response(
        self,
        response: Dict[str, Any],
        expected_fields: List[str]
    ) -> Optional[str]:
        """
        Validate API response format.
        
        Args:
            response (Dict[str, Any]): Response to validate
            expected_fields (List[str]): Required fields
            
        Returns:
            Optional[str]: Error message if validation fails, None if okay
        """
        try:
            # If the response itself has an error, short-circuit
            if "error" in response:
                return f"Response contains error: {response['error']}"
            
            # Check for required fields
            for field in expected_fields:
                if field not in response:
                    return f"Missing required field: {field}"
            
            # Must have "individual_statements"
            if "individual_statements" not in response:
                return "Missing individual_statements field"
            
            if not isinstance(response["individual_statements"], list):
                return "individual_statements must be a list"
            
            # Check each statement’s structure
            for stmt in response["individual_statements"]:
                if not isinstance(stmt, dict):
                    return "Each statement must be a dictionary"
                if not all(k in stmt for k in ["statement", "statement_id", "explanation"]):
                    return "Statement missing required fields"
            
            return None
        
        except Exception as e:
            logging.error(f"Response validation error: {str(e)}", exc_info=True)
            return f"Validation error: {str(e)}"

    def format_output(
        self,
        responses: Dict[str, Dict[str, Any]],
        article_id: str
    ) -> Dict[str, Any]:
        """
        Format chain responses into final output.
        
        Args:
            responses (Dict[str, Dict[str, Any]]): Chain responses
            article_id (str): Article identifier
            
        Returns:
            Dict[str, Any]: Formatted output
        """
        try:
            output = {
                "id": article_id,
                "individual_statements": [],
                "chain_responses": responses
            }
            
            # Aggregate all "individual_statements" from each response
            for phase, resp in responses.items():
                if "individual_statements" in resp:
                    output["individual_statements"].extend(resp["individual_statements"])
            
            return output
        
        except Exception as e:
            logging.error(
                f"Output formatting error for article {article_id}: {str(e)}",
                extra={"article_id": article_id},
                exc_info=True
            )
            return {
                "id": article_id,
                "error": f"Formatting error: {str(e)}",
                "individual_statements": []
            }
