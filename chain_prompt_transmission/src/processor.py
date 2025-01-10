#!/usr/bin/env python3
"""
processor.py
Implements the old multi-phase pipeline for article processing:
(extract → contextualize → temporal → mechanism → validate)
"""

from typing import Dict, Any, Tuple, Optional, List, Set
import logging
import os
import json
import yaml
import tempfile
from pathlib import Path
from enum import Enum
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler

# Local imports
from .api import APIClient
from .validators import InputValidator, ArticleSchema


##############################################################################
#                          Enums & Phase Tracking                            #
##############################################################################

class PhaseStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PhaseState:
    status: PhaseStatus = PhaseStatus.NOT_STARTED
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    statement_ids: Set[str] = field(default_factory=set)
    temp_files: Set[str] = field(default_factory=set)


class PhaseTracker:
    """
    Tracks the status of each phase, including any temporary files created
    or cleaned up in each phase. Also has an audit log to record steps.
    """
    def __init__(self):
        self.phases = {
            "extract": PhaseState(),
            "contextualize": PhaseState(),
            "temporal": PhaseState(),
            "mechanism": PhaseState(),
            "validate": PhaseState()
        }
        self.temp_dir = tempfile.mkdtemp()
        self.used_statement_ids: Set[str] = set()
        self.audit_trail = []

    def start_phase(self, phase: str):
        self.phases[phase].status = PhaseStatus.IN_PROGRESS
        self.add_audit_entry(
            phase=phase,
            action="start",
            status=PhaseStatus.IN_PROGRESS.value,
            details={"temp_files": list(self.phases[phase].temp_files)}
        )

    def complete_phase(self, phase: str, data: Dict[str, Any], statement_ids: Set[str]):
        state = self.phases[phase]
        state.status = PhaseStatus.COMPLETED
        state.data = data
        state.statement_ids = statement_ids
        self.used_statement_ids.update(statement_ids)

        self.add_audit_entry(
            phase=phase,
            action="complete",
            status=PhaseStatus.COMPLETED.value,
            details={"statement_ids": list(statement_ids)},
            statement_count=len(statement_ids)
        )

    def fail_phase(self, phase: str, error: str):
        state = self.phases[phase]
        state.status = PhaseStatus.FAILED
        state.error = error

        self.add_audit_entry(
            phase=phase,
            action="fail",
            status=PhaseStatus.FAILED.value,
            details={"temp_files": list(state.temp_files)},
            error=error
        )

        self.rollback_phase(phase)

    def rollback_phase(self, phase: str):
        state = self.phases[phase]
        for temp_file in state.temp_files:
            try:
                os.remove(temp_file)
            except OSError as e:
                self.add_audit_entry(
                    phase=phase,
                    action="cleanup_error",
                    status=state.status.value,
                    details={"file": temp_file},
                    error=str(e)
                )
        state.temp_files.clear()
        state.data = None

        self.add_audit_entry(
            phase=phase,
            action="rollback",
            status=state.status.value,
            details={"cleaned_files": True}
        )

    def add_audit_entry(self, **kwargs):
        self.audit_trail.append(kwargs)

    def get_audit_log(self) -> Dict[str, Any]:
        return {"audit_entries": self.audit_trail}

    def cleanup(self):
        """
        Called on script termination or signal, to remove leftover temp files.
        """
        try:
            for phase_state in self.phases.values():
                for temp_file in phase_state.temp_files:
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass
            os.rmdir(self.temp_dir)
        except Exception as e:
            logging.error(f"Cleanup error: {str(e)}", exc_info=True)


##############################################################################
#                     Article Logging & Metadata Classes                     #
##############################################################################

@dataclass
class ProcessingMetadata:
    """
    Tracks metadata about the processing of an article:
    - validation_changes
    - discarded_statements
    - statement_counts
    """
    validation_changes: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    discarded_statements: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    statement_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def total_statements(self) -> int:
        return sum(self.statement_counts.values())


class ArticleLogger:
    """
    A rotating file logger that also logs to the console.
    Used for detailed + error logs.
    """
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
        self._setup_logging()

    def _setup_logging(self):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        detailed_log = os.path.join(log_dir, 'processing_detailed.log')
        fh = RotatingFileHandler(
            detailed_log,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        error_log = os.path.join(log_dir, 'processing_errors.log')
        eh = RotatingFileHandler(
            error_log,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        eh.setLevel(logging.ERROR)
        eh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(eh)
        self.logger.addHandler(ch)
        self.logger.setLevel(logging.DEBUG)

    def log_phase_start(self, phase: str, article_id: str):
        self.logger.info(f"Starting {phase} phase for article {article_id}")

    def log_phase_completion(self, phase: str, article_id: str, data: Dict[str, Any]):
        self.logger.info(
            f"Completed {phase} phase for article {article_id}",
            extra={"data": data}
        )

    def log_phase_error(self, phase: str, article_id: str, error: str):
        self.logger.error(
            f"Error in {phase} phase for article {article_id}: {error}",
            extra={"article_id": article_id, "phase": phase}
        )

    def log_validation_changes(self, article_id: str, metadata: ProcessingMetadata):
        self.logger.info(
            f"Validation changes for article {article_id}",
            extra={"article_id": article_id, "changes": metadata.validation_changes}
        )

    def log_statement_counts(self, article_id: str, metadata: ProcessingMetadata):
        self.logger.info(
            f"Statement counts for article {article_id}",
            extra={"article_id": article_id, "counts": metadata.statement_counts}
        )


##############################################################################
#                      The Main ArticleProcessor Class                       #
##############################################################################

class ArticleProcessor:
    """
    Implements the old multi-phase pipeline:
        1) Extract
        2) Contextualize
        3) Temporal
        4) Mechanism
        5) Validate
    """
    def __init__(
        self,
        api_client: APIClient,
        validator: InputValidator,
        config_path: Path,
        prompts_path: Path
    ):
        print(f"Loading prompts from: {prompts_path.absolute()}")
        self.api_client = api_client
        self.validator = validator

        # Load YAML config & prompts
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        with open(prompts_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

        self.logger = ArticleLogger()
        self.phase_tracker = PhaseTracker()

    @contextmanager
    def open_output_file(self, filename: str, mode: str = 'w'):
        f = open(filename, mode, encoding='utf-8')
        try:
            yield f
        finally:
            f.close()

    def format_prompt(self, prompt_key: str, **kwargs) -> str:
        """
        Pull a prompt template from self.prompts and format with any kwargs.
        """
        try:
            prompt_template = self.prompts[prompt_key]["prompt"]
            return prompt_template.format(**kwargs)
        except KeyError as e:
            self.logger.logger.error(f"Prompt key not found: {prompt_key}")
            raise KeyError(f"Prompt key not found: {prompt_key}") from e

    def _transform_individual_statements(
        self,
        individual_statements: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Convert an array of statements into a flattened dict such that:
        {
            "affirm_transmissible_statement_extract_1": "<the statement text>",
            "affirm_transmissible_statement_extract_1_explanation": "<the explanation>"
        }
        """
        flattened = {}
        for item in individual_statements:
            # item["statement_id"] is our key
            statement_id = item["statement_id"]
            statement_text = item["statement"]
            explanation_text = item["explanation"]

            # Add two keys: statement_id, and statement_id + "_explanation"
            flattened[statement_id] = statement_text
            flattened[f"{statement_id}_explanation"] = explanation_text

        return flattened

    # --------------------- Phase: Extract ---------------------
    def process_extract_phase(
        self,
        article: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Phase that extracts base statements from the article.
        """
        article_id = article["id"]
        self.logger.log_phase_start("extract", str(article_id))

        try:
            prompt = self.format_prompt(
                "affirm_transmissible_extract",
                fulltext=article["fulltext"]
            )
            result = self.api_client.make_call(prompt, article_id)

            # Validate the response
            error = self.validator.validate_api_response(
                result,
                expected_fields=["statements", "individual_statements"],
                response_type="extract"
            )
            if error:
                self.logger.log_phase_error("extract", str(article_id), error)
                return None, error

            self.logger.log_phase_completion("extract", str(article_id), result)
            return {
                "statements": result["statements"],
                "individual_statements": result.get("individual_statements", [])
            }, None

        except Exception as e:
            msg = f"Extract phase failed for article {article_id}: {str(e)}"
            self.logger.logger.error(msg, exc_info=True)
            return None, str(e)

    # --------------------- Phase: Contextualize ---------------------
    def process_contextualize_phase(
        self,
        article: Dict[str, Any],
        statements: List[str]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        article_id = article["id"]
        self.logger.log_phase_start("contextualize", str(article_id))

        try:
            prompt = self.format_prompt(
                "affirm_transmissible_contextualize",
                fulltext=article["fulltext"],
                statements_list=json.dumps(statements)
            )
            result = self.api_client.make_call(prompt, article_id)

            error = self.validator.validate_api_response(
                result,
                expected_fields=["context_added", "unchanged", "all_statements", "individual_statements"],
                response_type="contextualize"
            )
            if error:
                self.logger.log_phase_error("contextualize", str(article_id), error)
                return None, error

            self.logger.log_phase_completion("contextualize", str(article_id), result)
            return result, None

        except Exception as e:
            msg = f"Contextualize phase failed for article {article_id}: {str(e)}"
            self.logger.logger.error(msg, exc_info=True)
            return None, str(e)

    # --------------------- Phase: Temporal ---------------------
    def process_temporal_phase(
        self,
        article: Dict[str, Any],
        statements: List[str]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        article_id = article["id"]
        self.logger.log_phase_start("temporal", str(article_id))

        try:
            prompt = self.format_prompt(
                "affirm_transmissible_exclude_1_temporal",
                all_statements=json.dumps(statements)
            )
            result = self.api_client.make_call(prompt, article_id)

            error = self.validator.validate_api_response(
                result,
                expected_fields=["statements", "individual_statements"],
                response_type="temporal"
            )
            if error:
                self.logger.log_phase_error("temporal", str(article_id), error)
                return None, error

            # If you have a specialized temporal validator
            # self.validator.validate_temporal_response(result)

            self.logger.log_phase_completion("temporal", str(article_id), result)
            return {
                "statements": result["statements"],
                "individual_statements": result.get("individual_statements", [])
            }, None

        except Exception as e:
            msg = f"Temporal phase failed for article {article_id}: {str(e)}"
            self.logger.logger.error(msg, exc_info=True)
            return None, str(e)

    # --------------------- Phase: Mechanism/Exclusion ---------------------
    def process_exclude_phases(
        self,
        article: Dict[str, Any],
        statements: List[str]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        article_id = article["id"]
        self.logger.log_phase_start("mechanism", str(article_id))

        try:
            # Temporal exclusion
            temporal_prompt = self.format_prompt(
                "affirm_transmissible_exclude_1_temporal",
                all_statements=json.dumps(statements)
            )
            temporal_result = self.api_client.make_call(temporal_prompt, article_id)
            if "error" in temporal_result:
                error_msg = "Error in temporal exclusion phase"
                self.logger.log_phase_error("mechanism", str(article_id), error_msg)
                return None, error_msg

            # Mechanism exclusion
            mechanism_prompt = self.format_prompt(
                "affirm_transmissible_exclude_2_blanket_v_mechanism",
                all_statements=json.dumps(temporal_result["statements"])
            )
            mechanism_result = self.api_client.make_call(mechanism_prompt, article_id)
            if "error" in mechanism_result:
                error_msg = "Error in mechanism exclusion phase"
                self.logger.log_phase_error("mechanism", str(article_id), error_msg)
                return None, error_msg

            # >>> REMOVE/COMMENT OUT these lines:
            # "temporal": {
            #     **temporal_flattened
            # },
            # "mechanism": {
            #     **mechanism_flattened
            # },

            # >>> AND INSTEAD use the raw structure:
            combined_result = {
                "temporal": {
                    "statements": temporal_result["statements"],
                    "individual_statements": temporal_result.get("individual_statements", [])
                },
                "mechanism": {
                    "mechanism_statements": mechanism_result.get("mechanism_statements", []),
                    "blanket_statements": mechanism_result.get("blanket_statements", []),
                    "individual_statements": mechanism_result.get("individual_statements", [])
                }
            }
            
            self.logger.log_phase_completion("mechanism", str(article_id), combined_result)
            return combined_result, None

        except Exception as e:
            msg = f"Exclusion phases failed for article {article_id}: {str(e)}"
            self.logger.logger.error(msg, exc_info=True)
            return None, str(e)

    # --------------------- Phase: Validate ---------------------
    def process_validate_phase(
        self,
        article: Dict[str, Any],
        mechanism_statements: List[str],
        blanket_statements: List[str]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        article_id = article["id"]
        self.logger.log_phase_start("validate", str(article_id))

        try:
            prompt = self.format_prompt(
                "affirm_transmissible_validate",
                fulltext=article["fulltext"],
                mechanism_statements=json.dumps(mechanism_statements),
                blanket_statements=json.dumps(blanket_statements)
            )
            result = self.api_client.make_call(prompt, article_id)

            if "error" in result:
                err_msg = "Error in validation phase"
                self.logger.log_phase_error("validate", str(article_id), err_msg)
                return None, err_msg

            error = self.validator.validate_api_response(
                result,
                expected_fields=[
                    "mechanism_validated",
                    "mechanism_discarded",
                    "blanket_validated",
                    "blanket_discarded",
                    "individual_statements"
                ],
                response_type="validate"
            )
            if error:
                self.logger.log_phase_error("validate", str(article_id), error)
                return None, error

            self.logger.log_phase_completion("validate", str(article_id), result)
            return result, None

        except Exception as e:
            msg = f"Validate phase failed for article {article_id}: {str(e)}"
            self.logger.logger.error(msg, exc_info=True)
            return None, str(e)

    # --------------------- Orchestrating All Phases ---------------------
    def process_article(
        self,
        article: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Runs the entire multi-phase pipeline on a single article.
        Returns final_data and discard_info (if any).
        """
        article_id = article["id"]
        self.logger.logger.info(f"Starting processing for article {article_id}")

        # Validate input structure
        error = self.validator.validate_article(article)
        if error:
            self.logger.logger.error(f"Validation failed for article {article_id}: {error}")
            return None, None

        try:
            # Extract Phase
            extract_result, error = self.process_extract_phase(article)
            if error or extract_result is None:
                return None, None

            # Contextualize Phase
            context_result, error = self.process_contextualize_phase(
                article,
                extract_result["statements"]
            )
            if error or context_result is None:
                return None, None

            # Temporal Phase
            temporal_result, error = self.process_temporal_phase(
                article,
                context_result["all_statements"]
            )
            if error or temporal_result is None:
                return None, None

            # Mechanism Phase
            mechanism_result, error = self.process_exclude_phases(
                article,
                temporal_result["statements"]
            )
            if error or mechanism_result is None:
                return None, None

            # Validation Phase
            validate_result, error = self.process_validate_phase(
                article,
                mechanism_result["mechanism"]["mechanism_statements"],
                mechanism_result["mechanism"]["blanket_statements"]
            )
            if error or validate_result is None:
                return None, None

            # Build metadata
            metadata = ProcessingMetadata(
                validation_changes={
                    "mechanism": [
                        {
                            "id": stmt["statement_id"],
                            "original": stmt.get("original", stmt["statement"]),
                            "final": stmt["statement"],
                            "location": stmt.get("location", ""),
                            "context": stmt.get("context", ""),
                            "explanation": stmt.get("explanation", "")
                        }
                        for stmt in validate_result["mechanism_validated"]
                    ],
                    "blanket": [
                        {
                            "id": stmt["statement_id"],
                            "original": stmt.get("original", stmt["statement"]),
                            "final": stmt["statement"],
                            "location": stmt.get("location", ""),
                            "context": stmt.get("context", ""),
                            "explanation": stmt.get("explanation", "")
                        }
                        for stmt in validate_result["blanket_validated"]
                    ]
                },
                discarded_statements={
                    "mechanism": [
                        {
                            "statement": stmt["statement"],
                            "reason": stmt["reason"],
                            "explanation": stmt["explanation"]
                        }
                        for stmt in validate_result["mechanism_discarded"]
                    ],
                    "blanket": [
                        {
                            "statement": stmt["statement"],
                            "reason": stmt["reason"],
                            "explanation": stmt["explanation"]
                        }
                        for stmt in validate_result["blanket_discarded"]
                    ]
                },
                statement_counts={
                    "extracted": len(extract_result["statements"]),
                    "contextualized": len(context_result["all_statements"]),
                    "temporal": len(temporal_result["statements"]),
                    "mechanism": len(mechanism_result["mechanism"]["mechanism_statements"]),
                    "blanket": len(mechanism_result["mechanism"]["blanket_statements"]),
                    "validated_mechanism": len(validate_result["mechanism_validated"]),
                    "validated_blanket": len(validate_result["blanket_validated"]),
                    "discarded_mechanism": len(validate_result["mechanism_discarded"]),
                    "discarded_blanket": len(validate_result["blanket_discarded"])
                }
            )

            # Build final output
            # 1) Flatten each phase’s individual statements
            extracted_flattened = self._transform_individual_statements(
                extract_result.get("individual_statements", [])
            )
            contextualized_flattened = self._transform_individual_statements(
                context_result.get("individual_statements", [])
            )
            temporal_flattened = self._transform_individual_statements(
                temporal_result.get("individual_statements", [])
            )
            mechanism_flattened = self._transform_individual_statements(
                mechanism_result["mechanism"].get("individual_statements", [])
            )
            validation_flattened = self._transform_individual_statements(
                validate_result.get("individual_statements", [])
            )

            # 2) Build final_data while excluding the old statement arrays
            final_data = {
                "id": article_id,
                "processing_stages": {
                    "extract": {
                        # Only the flattened statements, no raw arrays
                        **extracted_flattened
                    },
                    "contextualize": {
#                        "context_added": context_result["context_added"],
#                        "unchanged": context_result["unchanged"],
                        **contextualized_flattened
                    },
                    "temporal": {
                        **temporal_flattened
                    },
                    "mechanism": {
                        **mechanism_flattened
                    },
                    "validation": {
                        # If you do NOT want validated/discarded lists, don’t include them here
                        # "mechanism_validated": validate_result["mechanism_validated"],
                        # "mechanism_discarded": validate_result["mechanism_discarded"],
                        # "blanket_validated": validate_result["blanket_validated"],
                        # "blanket_discarded": validate_result["blanket_discarded"],
                        **validation_flattened
                    }
                },
                # Keep validation_metadata if you want. Or remove it if it references statements
                "validation_metadata": metadata.__dict__
            }

            # Optionally keep or remove the statement_counts inside metadata if they are referencing
            # the old lists. But they won't break anything if you keep them.

            # Optionally add some counts relevant to your logic (demo only):
            final_data.update({
                "affirm_transmissible_count": 0,  # As relevant for your pipeline
                "affirm_why_count": 0,
                "affirm_keywords_count": 0,
                "deny_transmissible_count": 0,
                "deny_why_count": 0,
                "deny_keywords_count": 0
            })

            # Discard info if anything was actually discarded
            discard_info = None
            if validate_result["mechanism_discarded"] or validate_result["blanket_discarded"]:
                discard_info = {
                    "id": article_id,
                    "fulltext": article["fulltext"],
                    "discarded_statements": metadata.discarded_statements
                }

            self.logger.logger.info(
                f"Successfully processed article {article_id}",
                extra={"article_id": article_id, "statement_counts": metadata.statement_counts}
            )

            return final_data, discard_info

        except Exception as e:
            msg = f"Processing failed for article {article_id}: {str(e)}"
            self.logger.logger.error(msg, exc_info=True)
            return None, None
