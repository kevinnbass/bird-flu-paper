from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import json
from pathlib import Path
import logging

@dataclass
class ArticleSchema:
    """
    Defines the expected structure and constraints for input articles.
    
    Attributes:
        required_fields (List[str]): List of fields that must be present in each article
        id_type (str): Expected type for the id field
        fulltext_type (str): Expected type for the fulltext field
        id_range (Dict[str, int]): Valid range for article IDs
    """
    required_fields: List[str] = None
    id_type: str = "string"
    fulltext_type: str = "string"
    id_range: Dict[str, int] = None

    def __post_init__(self):
        """Initialize default values if none provided."""
        if self.required_fields is None:
            self.required_fields = ["id", "fulltext"]
        if self.id_range is None:
            self.id_range = {"min": 475, "max": 485}

class InputValidator:
    """Validates input data against defined schemas and constraints."""
    
    def __init__(self, schema: ArticleSchema):
        """
        Initialize validator with schema.
        
        Args:
            schema (ArticleSchema): Schema defining validation rules
        """
        self.schema = schema
        
    def validate_article(self, article: Dict[str, Any]) -> Optional[str]:
        """
        Validate an article against the schema.
        
        Args:
            article (Dict[str, Any]): Article to validate
            
        Returns:
            Optional[str]: Error message if validation fails, None if successful
        """
        try:
            # Check for required fields
            for field in self.schema.required_fields:
                if field not in article:
                    return f"Missing required field: {field}"
                    
            # Validate ID format and range
            try:
                article_id = int(article["id"])
                if not (self.schema.id_range["min"] <= article_id <= self.schema.id_range["max"]):
                    return (f"Article ID {article_id} outside valid range "
                           f"({self.schema.id_range['min']}-{self.schema.id_range['max']})")
            except ValueError:
                return f"Invalid article ID format: {article['id']}"
                
            # Validate fulltext
            if not isinstance(article["fulltext"], str):
                return "Fulltext must be a string"
            if not article["fulltext"].strip():
                return "Fulltext cannot be empty"
                
            return None
            
        except Exception as e:
            logging.error(f"Validation error: {str(e)}", exc_info=True)
            return f"Validation error: {str(e)}"
    
    def validate_api_response(
        self, 
        response: Dict[str, Any], 
        expected_fields: List[str],
        response_type: Optional[str] = None
    ) -> Optional[str]:
        """
        Validate API response format and content.
        
        Args:
            response (Dict[str, Any]): Response to validate
            expected_fields (List[str]): Required fields
            response_type (Optional[str]): Type of response for specific validations
            
        Returns:
            Optional[str]: Error message if validation fails, None if successful
        """
        try:
            # Check for error field
            if "error" in response:
                return f"API returned error: {response['error']}"

            # Check for expected fields
            for field in expected_fields:
                if field not in response:
                    return f"Missing expected field in API response: {field}"
            
            # Validate individual statements if present
            if "individual_statements" not in response:
                return "Missing individual_statements in API response"
                
            for stmt in response["individual_statements"]:
                error = self.validate_individual_statement(stmt, response_type)
                if error:
                    return error
            
            # Type-specific validations
            if response_type:
                error = self.validate_response_type(response, response_type)
                if error:
                    return error
            
            return None
            
        except Exception as e:
            logging.error(f"API response validation error: {str(e)}", exc_info=True)
            return f"API response validation error: {str(e)}"
            
    def validate_individual_statement(
        self,
        stmt: Dict[str, Any],
        response_type: Optional[str]
    ) -> Optional[str]:
        """
        Validate an individual statement.
        
        Args:
            stmt (Dict[str, Any]): Statement to validate
            response_type (Optional[str]): Type of response for specific validations
            
        Returns:
            Optional[str]: Error message if validation fails, None if successful
        """
        try:
            if not isinstance(stmt, dict):
                return "Individual statement must be a dictionary"
                
            required_fields = ["statement", "statement_id", "explanation"]
            for field in required_fields:
                if field not in stmt:
                    return f"Individual statement missing '{field}' field"
                    
            # Validate statement content
            if not isinstance(stmt["statement"], str):
                return "Statement must be a string"
            if not stmt["statement"].strip():
                return "Statement cannot be empty"
                
            # Validate statement_id format
            if response_type:
                valid_prefixes = {
                    "affirm_transmissible": ["affirm_transmissible_statement_"],
                    "affirm_why": ["affirm_why_statement_"],
                    "affirm_keywords": ["affirm_keywords_statement_"],
                    "deny_transmissible": ["deny_transmissible_statement_"],
                    "deny_why": ["deny_why_statement_"],
                    "deny_keywords": ["deny_keywords_statement_"]
                }
                
                if response_type in valid_prefixes:
                    if not any(stmt["statement_id"].startswith(prefix) 
                             for prefix in valid_prefixes[response_type]):
                        return f"Invalid statement_id prefix for type {response_type}"
                        
                # Validate statement ID numbering
                try:
                    suffix = stmt["statement_id"].split("_")[-1]
                    if not suffix.isdigit():
                        return f"Statement ID suffix must be numeric, got '{suffix}'"
                    num = int(suffix)
                    if num < 1:
                        return "Statement ID numbering must start from 1"
                except (IndexError, ValueError):
                    return f"Invalid statement_id format: {stmt['statement_id']}"
                    
            return None
            
        except Exception as e:
            logging.error(f"Individual statement validation error: {str(e)}", exc_info=True)
            return str(e)
            
    def validate_response_type(
        self,
        response: Dict[str, Any],
        response_type: str
    ) -> Optional[str]:
        """
        Validate type-specific aspects of the response.
        
        Args:
            response (Dict[str, Any]): Response to validate
            response_type (str): Type of response for specific validations
            
        Returns:
            Optional[str]: Error message if validation fails, None if successful
        """
        try:
            if response_type == "affirm_transmissible":
                if not isinstance(response.get("statements", []), list):
                    return "Statements must be a list"
                    
            elif response_type == "affirm_why":
                if not isinstance(response.get("mechanism_statements", []), list):
                    return "Mechanism statements must be a list"
                if not isinstance(response.get("blanket_statements", []), list):
                    return "Blanket statements must be a list"
                    
            elif response_type.startswith(("affirm_keywords", "deny_keywords")):
                for stmt in response.get("individual_statements", []):
                    text = stmt["statement"].lower()
                    keywords = {"pandemic", "pandemics", "coronavirus", "covid", "covid-19"}
                    if not any(keyword in text for keyword in keywords):
                        return f"Statement missing required keyword: {stmt['statement']}"
                        
            elif response_type.startswith("deny_"):
                # Additional validation for deny-type responses could be added here
                pass
                
            return None
            
        except Exception as e:
            logging.error(f"Response type validation error: {str(e)}", exc_info=True)
            return str(e)
            
    def validate_temporal_response(self, response: Dict[str, Any]) -> Optional[str]:
        """
        Validate temporal phase response schema.
        
        Args:
            response (Dict[str, Any]): Response to validate
            
        Returns:
            Optional[str]: Error message if validation fails, None if successful
        """
        try:
            if not isinstance(response.get("statements", []), list):
                return "Statements must be a list"
                
            for stmt in response.get("individual_statements", []):
                if not isinstance(stmt.get("statement"), str):
                    return "Statement must be a string"
                if not stmt.get("statement", "").strip():
                    return "Statement cannot be empty"
                    
                # Validate temporal criteria
                text = stmt["statement"].lower()
                if any(word in text for word in ["current", "currently", "now", "present"]):
                    return f"Statement contains present-tense indicators: {stmt['statement']}"
                    
                if not any(word in text for word in ["would", "could", "might", "may", "future", "potential"]):
                    return f"Statement lacks future/potential indicators: {stmt['statement']}"
                    
            return None
            
        except Exception as e:
            logging.error(f"Temporal validation error: {str(e)}", exc_info=True)
            return str(e)
            
    def validate_jsonl_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Validate all articles in a JSONL file.
        
        Args:
            file_path (Path): Path to the JSONL file
            
        Returns:
            List[Dict[str, Any]]: List of validation errors with article IDs
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            JSONDecodeError: If the file contains invalid JSON
        """
        validation_errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    try:
                        article = json.loads(line.strip())
                        error = self.validate_article(article)
                        
                        if error:
                            validation_errors.append({
                                "line": line_number,
                                "article_id": article.get("id", "unknown"),
                                "error": error
                            })
                            
                    except json.JSONDecodeError as e:
                        validation_errors.append({
                            "line": line_number,
                            "article_id": "unknown",
                            "error": f"Invalid JSON: {str(e)}"
                        })
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {file_path}")
            
        return validation_errors