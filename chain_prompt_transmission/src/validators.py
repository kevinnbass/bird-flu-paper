from typing import Dict, Any, List, Optional
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
            self.id_range = {"min": 0, "max": 2033}


class InputValidator:
    """
    Validates input data (articles) and API responses for each phase:
      - extract
      - contextualize
      - trim
      - merge
      - temporal
      - remainder
      - mechanism
      - validate
    """

    def __init__(self, schema: ArticleSchema):
        """
        Initialize validator with the given schema.
        """
        self.schema = schema
        
    ########################################################################
    #           1) ARTICLE-LEVEL VALIDATION (validate_article)             #
    ########################################################################

    def validate_article(self, article: Dict[str, Any]) -> Optional[str]:
        """
        Validate the structure of an input article against ArticleSchema.
        
        Returns an error string if invalid, else None.
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
    
    ########################################################################
    #           2) API RESPONSE VALIDATION (validate_api_response)         #
    ########################################################################

    def validate_api_response(
        self, 
        response: Dict[str, Any], 
        expected_fields: List[str],
        response_type: Optional[str] = None
    ) -> Optional[str]:
        """
        Validate an API response (JSON) from any phase.

        expected_fields: required top-level fields in the JSON
        response_type: used to do phase-specific validations, if desired
        """
        try:
            # Check for explicit "error"
            if "error" in response:
                return f"API returned error: {response['error']}"

            # Check for required fields
            for field in expected_fields:
                if field not in response:
                    return f"Missing expected field in API response: {field}"
            
            # Check for 'individual_statements'
            if "individual_statements" not in response:
                return "Missing individual_statements in API response"
                
            # Validate each statement
            for stmt in response["individual_statements"]:
                error = self.validate_individual_statement(stmt, response_type)
                if error:
                    return error
            
            # If a phase is specified, do specialized checks
            if response_type:
                # E.g. temporal check
                if response_type == "temporal":
                    error = self.validate_temporal_response(response)
                    if error:
                        return error
                # E.g. remainder check
                elif response_type == "remainder":
                    error = self.validate_remainder_response(response)
                    if error:
                        return error
                else:
                    # General fallback
                    error = self.validate_response_type(response, response_type)
                    if error:
                        return error
            
            return None
            
        except Exception as e:
            logging.error(f"API response validation error: {str(e)}", exc_info=True)
            return f"API response validation error: {str(e)}"
    
    ########################################################################
    #           3) INDIVIDUAL STATEMENT VALIDATION                         #
    ########################################################################

    def validate_individual_statement(
        self,
        stmt: Dict[str, Any],
        response_type: Optional[str]
    ) -> Optional[str]:
        """
        Validate structure/format of a single 'statement' object in 'individual_statements'.
        """
        try:
            if not isinstance(stmt, dict):
                return "Individual statement must be a dictionary"

            # These fields are mandatory:
            required_fields = ["statement", "statement_id"]
            for f in required_fields:
                if f not in stmt:
                    return f"Individual statement missing '{f}' field"

            # statement must be non-empty
            if not isinstance(stmt["statement"], str):
                return "Statement must be a string"
            if not stmt["statement"].strip():
                return "Statement cannot be empty"

            # If desired, do prefix checks for each phase
            # e.g. "affirm_transmissible_statement_exclude_remainder_..."
            if response_type == "remainder":
                if not stmt["statement_id"].startswith("affirm_transmissible_statement_exclude_remainder_"):
                    return f"Statement ID not remainder prefix: {stmt['statement_id']}"
            
            elif response_type == "temporal":
                if not stmt["statement_id"].startswith("affirm_transmissible_statement_exclude_1_temporal_"):
                    # If you're strict about naming for temporal
                    return f"Statement ID not temporal prefix: {stmt['statement_id']}"
            
            # Always check numeric suffix, e.g. _3
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
    
    ########################################################################
    #           4) TYPE-SPECIFIC VALIDATION (validate_response_type)        #
    ########################################################################

    def validate_response_type(
        self,
        response: Dict[str, Any],
        response_type: str
    ) -> Optional[str]:
        """
        Validate top-level structure or fields for a given response_type.
        Example: "affirm_transmissible", "validate", etc.
        """
        try:
            if response_type == "affirm_transmissible":
                # e.g. check 'statements' is a list
                if not isinstance(response.get("statements", []), list):
                    return "Statements must be a list"

            elif response_type == "validate":
                # ensure we have the final validated/discarded fields
                needed_keys = [
                    "mechanism_validated", 
                    "mechanism_discarded",
                    "blanket_validated", 
                    "blanket_discarded"
                ]
                for k in needed_keys:
                    if k not in response:
                        return f"Missing field '{k}' in validation response"

            # fallback: no special checks
            return None
        
        except Exception as e:
            logging.error(f"Response type validation error: {str(e)}", exc_info=True)
            return str(e)

    ########################################################################
    #           5) SPECIALIZED: validate_temporal_response                 #
    ########################################################################

    def validate_temporal_response(self, response: Dict[str, Any]) -> Optional[str]:
        """
        Minimal checks for the temporal phase.
        We do not override or re-filter LLM decisions here.
        """
        try:
            # Ensure "statements" is a list
            if not isinstance(response.get("statements", []), list):
                return "Statements must be a list"

            # Ensure each statement is a non-empty string
            for stmt in response.get("individual_statements", []):
                if not isinstance(stmt.get("statement"), str):
                    return "Statement must be a string"
                if not stmt["statement"].strip():
                    return "Statement cannot be empty"

            # No further logic: let the LLM handle everything
            return None

        except Exception as e:
            logging.error(f"Temporal validation error: {str(e)}", exc_info=True)
            return str(e)

    ########################################################################
    #           6) SPECIALIZED: validate_remainder_response                #
    ########################################################################

    def validate_remainder_response(self, response: Dict[str, Any]) -> Optional[str]:
        """
        Extra checks for the remainder phase. This is optional—add or remove
        any logic you want for 'exclude_remainder' type responses.
        """
        try:
            # e.g. check that "statements" is a list
            if not isinstance(response.get("statements", []), list):
                return "Statements must be a list"

            for stmt in response.get("individual_statements", []):
                # Basic checks
                if not isinstance(stmt.get("statement"), str):
                    return "Statement must be a string"
                if not stmt["statement"].strip():
                    return "Statement cannot be empty"
                # Potential remainder-specific logic could go here
                # e.g. checking that statements mention "x" or do not mention "y"

            return None
        except Exception as e:
            logging.error(f"Remainder validation error: {str(e)}", exc_info=True)
            return str(e)

    ########################################################################
    #           7) JSONL FILE VALIDATION (validate_jsonl_file)             #
    ########################################################################

    def validate_jsonl_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Validate all articles in a JSONL file for correct structure.

        Returns a list of errors with line numbers and article IDs, if any.
        """
        validation_errors = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    if not line.strip():
                        continue
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
