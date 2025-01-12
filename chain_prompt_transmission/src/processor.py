#!/usr/bin/env python3
"""
processor.py
Implements the multi-phase pipeline for article processing:
(extract → contextualize → trim → merge → temporal → remainder → mechanism → validate)
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
        # Notice the new "remainder" phase inserted after "temporal"
        self.phases = {
            "extract": PhaseState(),
            "contextualize": PhaseState(),
            "trim": PhaseState(),
            "merge": PhaseState(),
            "temporal": PhaseState(),
            "remainder": PhaseState(),   # NEW
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
#                      Logging & Metadata Classes                            #
##############################################################################

@dataclass
class ProcessingMetadata:
    """
    Tracks metadata about the processing of an article.
    Explanation fields are omitted to keep them out of final output.
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
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        error_log = os.path.join(log_dir, 'processing_errors.log')
        eh = RotatingFileHandler(
            error_log,
            maxBytes=5 * 1024 * 1024,
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


##############################################################################
#                          JSONL Append Helper                               #
##############################################################################

def append_jsonl(record: Dict[str, Any], filename: str):
    """
    Helper function to append a record to a JSONL file.
    Used for logging excluded statements from trim/temporal/remainder, etc.
    """
    with open(filename, 'a', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)
        f.write('\n')


##############################################################################
#                    The Main ArticleProcessor Class                        #
##############################################################################

class ArticleProcessor:
    """
    Implements the multi-phase pipeline:
      1) Extract
      2) Contextualize
      3) Trim
      4) Merge
      5) Temporal
      6) Remainder
      7) Mechanism
      8) Validate
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

        # Load config & prompts
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        with open(prompts_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

        self.logger = ArticleLogger()
        self.phase_tracker = PhaseTracker()

        ################################################################
        # Setup separate JSONL files for each filter's excluded statements
        # (These keys must exist in config.yaml under files: output:)
        ################################################################
        outfiles = self.config["files"]["output"]
        self.trim_excluded_file = outfiles.get("excluded_trim", "excluded_trim.jsonl")
        self.temporal_excluded_file = outfiles.get("excluded_temporal", "excluded_temporal.jsonl")
        self.remainder_excluded_file = outfiles.get("excluded_remainder", "excluded_remainder.jsonl")

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
            template = self.prompts[prompt_key]["prompt"]
            return template.format(**kwargs)
        except KeyError as e:
            self.logger.logger.error(f"Prompt key not found: {prompt_key}")
            raise KeyError(f"Prompt key not found: {prompt_key}") from e

    ############################################################################
    #                          HELPER FUNCTIONS                                #
    ############################################################################

    def _transform_individual_statements(
        self,
        individual_statements: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Convert array of statements into a flattened dict of {id: text}.
        """
        flattened = {}
        for item in individual_statements:
            statement_id = item["statement_id"]
            statement_text = item["statement"]
            flattened[statement_id] = statement_text
        return flattened

    def _pair_contextualize_output(
        self,
        extract_result: Dict[str, Any],
        context_result: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Build a side-by-side list comparing extract statements vs contextualized statements.
        """
        extract_map = {
            d["statement_id"]: d["statement"]
            for d in extract_result.get("individual_statements", [])
        }
        contextual_map = {}
        for d in context_result.get("individual_statements", []):
            c_id = d["statement_id"]
            paired_id = c_id.replace("contextualize_", "extract_")
            contextual_map[paired_id] = d["statement"]

        output_list = []
        for d in extract_result.get("individual_statements", []):
            e_id = d["statement_id"]
            c_id = e_id.replace("extract_", "contextualize_")
            e_text = extract_map[e_id]
            c_text = contextual_map.get(e_id, e_text)
            output_list.append({e_id: e_text, c_id: c_text})
        return output_list

    def _log_excluded_statements(
        self,
        input_stmts: List[str],
        output_stmts: List[str],
        phase_name: str,
        article_id: str,
        filename: str
    ):
        """
        Compare pre-filter statements to post-filter statements to see which got excluded.
        Then append them to a separate JSONL for review.
        """
        input_set = set(input_stmts)
        output_set = set(output_stmts)
        excluded = list(input_set - output_set)
        if excluded:
            record = {
                "article_id": article_id,
                "phase": phase_name,
                "excluded_statements": excluded
            }
            append_jsonl(record, filename)

    ############################################################################
    #               ERROR-HANDLING UTILITY: SERIOUS VS NON-SERIOUS             #
    ############################################################################

    def _handle_validation_error(
        self,
        phase: str,
        article_id: str,
        error: str
    ) -> Optional[Dict[str, Any]]:
        """
        Decide if an error is 'serious' (must abort) or not. If non-serious,
        log a warning and return an empty fallback result. Otherwise, fail phase.
        
        Returns:
          - None if we called fail_phase (serious error, must abort).
          - A minimal dict if we consider it non-serious and want to proceed.
        """
        # Define some keywords that indicate a SERIOUS error:
        SERIOUS_KEYWORDS = [
            "JSON parse error",
            "Missing required field",
            "Response contains error",
            "API error",
            "Validation error",
            "KeyError",
            "ParseException",
        ]
        if any(kw.lower() in error.lower() for kw in SERIOUS_KEYWORDS):
            # Hard fail
            self.logger.log_phase_error(phase, str(article_id), error)
            self.phase_tracker.fail_phase(phase, error)
            return None
        else:
            # Soft fail -> log warning, proceed with empty result
            self.logger.logger.warning(
                f"Non-serious validation issue in {phase} phase for article {article_id}: {error}"
            )
            # Return an empty "statements" object so we can keep going
            return {"statements": [], "individual_statements": []}

    ############################################################################
    #                          PHASE METHODS                                   #
    ############################################################################

    # --------------------- (1) Extract ---------------------
    def process_extract_phase(
        self,
        article: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        phase = "extract"
        article_id = article["id"]
        self.logger.log_phase_start(phase, str(article_id))
        self.phase_tracker.start_phase(phase)

        try:
            prompt = self.format_prompt(
                "affirm_transmissible_extract",
                fulltext=article["fulltext"]
            )
            result = self.api_client.make_call(prompt, article_id)

            error = self.validator.validate_api_response(
                result,
                expected_fields=["statements", "individual_statements"],
                response_type="extract"
            )
            if error:
                fallback = self._handle_validation_error(phase, str(article_id), error)
                if fallback is None:
                    # That means we aborted
                    return None, error
                # else treat fallback as result
                result = fallback

            self.logger.log_phase_completion(phase, str(article_id), result)
            statement_ids = {s["statement_id"] for s in result.get("individual_statements", [])}
            self.phase_tracker.complete_phase(phase, result, statement_ids)

            return {
                "statements": result.get("statements", []),
                "individual_statements": result.get("individual_statements", [])
            }, None

        except Exception as e:
            msg = f"{phase.capitalize()} phase failed for article {article_id}: {str(e)}"
            self.logger.log_phase_error(phase, str(article_id), msg)
            self.phase_tracker.fail_phase(phase, str(e))
            return None, str(e)

    # --------------------- (2) Contextualize ---------------------
    def process_contextualize_phase(
        self,
        article: Dict[str, Any],
        statements: List[str]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        phase = "contextualize"
        article_id = article["id"]
        self.logger.log_phase_start(phase, str(article_id))
        self.phase_tracker.start_phase(phase)

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
                fallback = self._handle_validation_error(phase, str(article_id), error)
                if fallback is None:
                    return None, error
                result = fallback

            self.logger.log_phase_completion(phase, str(article_id), result)
            statement_ids = {s["statement_id"] for s in result.get("individual_statements", [])}
            self.phase_tracker.complete_phase(phase, result, statement_ids)

            return result, None

        except Exception as e:
            msg = f"{phase.capitalize()} phase failed for article {article_id}: {str(e)}"
            self.logger.log_phase_error(phase, str(article_id), msg)
            self.phase_tracker.fail_phase(phase, str(e))
            return None, str(e)

    # --------------------- (3) Trim ---------------------
    def process_trim_phase(
        self,
        article: Dict[str, Any],
        original_extract: List[str],
        context_added: List[str]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        phase = "trim"
        article_id = article["id"]
        self.logger.log_phase_start(phase, str(article_id))
        self.phase_tracker.start_phase(phase)

        try:
            prompt = self.format_prompt(
                "affirm_transmissible_trim",
                fulltext=article["fulltext"],
                original_extract=json.dumps(original_extract),
                context_added=json.dumps(context_added)
            )
            result = self.api_client.make_call(prompt, article_id)

            error = self.validator.validate_api_response(
                result,
                expected_fields=["statements", "individual_statements"],
                response_type=None
            )
            if error:
                fallback = self._handle_validation_error(phase, str(article_id), error)
                if fallback is None:
                    return None, error
                result = fallback

            # Compare to see which got trimmed
            combined_input = original_extract + context_added
            output_stmts = result.get("statements", [])
            self._log_excluded_statements(
                input_stmts=combined_input,
                output_stmts=output_stmts,
                phase_name=phase,
                article_id=str(article_id),
                filename=self.trim_excluded_file
            )

            self.logger.log_phase_completion(phase, str(article_id), result)
            statement_ids = {s["statement_id"] for s in result.get("individual_statements", [])}
            self.phase_tracker.complete_phase(phase, result, statement_ids)

            return result, None

        except Exception as e:
            msg = f"{phase.capitalize()} phase failed for article {article_id}: {str(e)}"
            self.logger.log_phase_error(phase, str(article_id), msg)
            self.phase_tracker.fail_phase(phase, str(e))
            return None, str(e)

    # --------------------- (4) Merge ---------------------
    def process_merge_phase(
        self,
        article: Dict[str, Any],
        trimmed_statements: List[str]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        phase = "merge"
        article_id = article["id"]
        self.logger.log_phase_start(phase, str(article_id))
        self.phase_tracker.start_phase(phase)

        try:
            prompt = self.format_prompt(
                "affirm_transmissible_merge",
                fulltext=article["fulltext"],
                trimmed_statements=json.dumps(trimmed_statements)
            )
            result = self.api_client.make_call(prompt, article_id)

            error = self.validator.validate_api_response(
                result,
                expected_fields=["statements", "individual_statements"],
                response_type=None
            )
            if error:
                fallback = self._handle_validation_error(phase, str(article_id), error)
                if fallback is None:
                    return None, error
                result = fallback

            self.logger.log_phase_completion(phase, str(article_id), result)
            statement_ids = {s["statement_id"] for s in result.get("individual_statements", [])}
            self.phase_tracker.complete_phase(phase, result, statement_ids)

            return result, None

        except Exception as e:
            msg = f"{phase.capitalize()} phase failed for article {article_id}: {str(e)}"
            self.logger.log_phase_error(phase, str(article_id), msg)
            self.phase_tracker.fail_phase(phase, str(e))
            return None, str(e)

    # --------------------- (5) Temporal ---------------------
    def process_temporal_phase(
        self,
        article: Dict[str, Any],
        statements: List[str]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        phase = "temporal"
        article_id = article["id"]
        self.logger.log_phase_start(phase, str(article_id))
        self.phase_tracker.start_phase(phase)

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
                fallback = self._handle_validation_error(phase, str(article_id), error)
                if fallback is None:
                    return None, error
                result = fallback

            # Log excluded statements
            output_stmts = result.get("statements", [])
            self._log_excluded_statements(
                input_stmts=statements,
                output_stmts=output_stmts,
                phase_name=phase,
                article_id=str(article_id),
                filename=self.temporal_excluded_file
            )

            self.logger.log_phase_completion(phase, str(article_id), result)
            statement_ids = {s["statement_id"] for s in result.get("individual_statements", [])}
            self.phase_tracker.complete_phase(phase, result, statement_ids)

            return {
                "statements": output_stmts,
                "individual_statements": result.get("individual_statements", [])
            }, None

        except Exception as e:
            msg = f"{phase.capitalize()} phase failed for article {article_id}: {str(e)}"
            self.logger.log_phase_error(phase, str(article_id), msg)
            self.phase_tracker.fail_phase(phase, str(e))
            return None, str(e)

    # --------------------- (6) Remainder (NEW) ---------------------
    def process_remainder_phase(
        self,
        article: Dict[str, Any],
        statements: List[str]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        phase = "remainder"
        article_id = article["id"]
        self.logger.log_phase_start(phase, str(article_id))
        self.phase_tracker.start_phase(phase)

        try:
            prompt = self.format_prompt(
                "affirm_transmissible_exclude_remainder",
                all_statements=json.dumps(statements)
            )
            result = self.api_client.make_call(prompt, article_id)

            error = self.validator.validate_api_response(
                result,
                expected_fields=["statements", "individual_statements"],
                response_type=None
            )
            if error:
                fallback = self._handle_validation_error(phase, str(article_id), error)
                if fallback is None:
                    return None, error
                result = fallback

            # Log excluded statements from remainder filter
            output_stmts = result.get("statements", [])
            self._log_excluded_statements(
                input_stmts=statements,
                output_stmts=output_stmts,
                phase_name=phase,
                article_id=str(article_id),
                filename=self.remainder_excluded_file
            )

            self.logger.log_phase_completion(phase, str(article_id), result)
            statement_ids = {s["statement_id"] for s in result.get("individual_statements", [])}
            self.phase_tracker.complete_phase(phase, result, statement_ids)

            return {
                "statements": output_stmts,
                "individual_statements": result.get("individual_statements", [])
            }, None

        except Exception as e:
            msg = f"{phase.capitalize()} phase failed for article {article_id}: {str(e)}"
            self.logger.log_phase_error(phase, str(article_id), msg)
            self.phase_tracker.fail_phase(phase, str(e))
            return None, str(e)

    # --------------------- (7) Mechanism/Blanket Exclusion ---------------------
    def process_exclude_phases(
        self,
        article: Dict[str, Any],
        statements: List[str]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        phase = "mechanism"
        article_id = article["id"]
        self.logger.log_phase_start(phase, str(article_id))
        self.phase_tracker.start_phase(phase)

        try:
            prompt = self.format_prompt(
                "affirm_transmissible_exclude_2_blanket_v_mechanism",
                all_statements=json.dumps(statements)
            )
            mechanism_result = self.api_client.make_call(prompt, article_id)

            if "error" in mechanism_result:
                # This indicates a top-level error from the LLM
                err_msg = "Response contains error (mechanism_result)"
                fallback = self._handle_validation_error(phase, str(article_id), err_msg)
                if fallback is None:
                    return None, err_msg
                mechanism_result = fallback

            error = self.validator.validate_api_response(
                mechanism_result,
                expected_fields=["mechanism_statements", "blanket_statements", "individual_statements"],
                response_type=None
            )
            if error:
                fallback = self._handle_validation_error(phase, str(article_id), error)
                if fallback is None:
                    return None, error
                mechanism_result = fallback

            self.logger.log_phase_completion(phase, str(article_id), mechanism_result)
            statement_ids = {s["statement_id"] for s in mechanism_result.get("individual_statements", [])}
            self.phase_tracker.complete_phase(phase, mechanism_result, statement_ids)

            # Rebuild final structure
            combined_result = {
                "mechanism": {
                    "mechanism_statements": mechanism_result.get("mechanism_statements", []),
                    "blanket_statements": mechanism_result.get("blanket_statements", []),
                    "individual_statements": mechanism_result.get("individual_statements", [])
                }
            }
            return combined_result, None

        except Exception as e:
            msg = f"Exclusion phases failed for article {article_id}: {str(e)}"
            self.logger.log_phase_error(phase, str(article_id), msg)
            self.phase_tracker.fail_phase(phase, str(e))
            return None, str(e)

    # --------------------- (8) Validate ---------------------
    def process_validate_phase(
        self,
        article: Dict[str, Any],
        mechanism_statements: List[str],
        blanket_statements: List[str]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        phase = "validate"
        article_id = article["id"]
        self.logger.log_phase_start(phase, str(article_id))
        self.phase_tracker.start_phase(phase)

        try:
            prompt = self.format_prompt(
                "affirm_transmissible_validate",
                fulltext=article["fulltext"],
                mechanism_statements=json.dumps(mechanism_statements),
                blanket_statements=json.dumps(blanket_statements)
            )
            result = self.api_client.make_call(prompt, article_id)

            if "error" in result:
                err_msg = "Response contains error (validate)"
                fallback = self._handle_validation_error(phase, str(article_id), err_msg)
                if fallback is None:
                    return None, err_msg
                result = fallback

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
                fallback = self._handle_validation_error(phase, str(article_id), error)
                if fallback is None:
                    return None, error
                result = fallback

            self.logger.log_phase_completion(phase, str(article_id), result)
            statement_ids = {s["statement_id"] for s in result.get("individual_statements", [])}
            self.phase_tracker.complete_phase(phase, result, statement_ids)

            return result, None

        except Exception as e:
            msg = f"Validate phase failed for article {article_id}: {str(e)}"
            self.logger.log_phase_error(phase, str(article_id), msg)
            self.phase_tracker.fail_phase(phase, str(e))
            return None, str(e)

    ############################################################################
    #                     Orchestrating ALL 8 Phases                           #
    ############################################################################
    def process_article(
        self,
        article: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Runs the entire multi-phase pipeline on a single article.
        Returns (final_data, discard_info) if successful, or (None, None) on error.
        """
        article_id = article["id"]
        self.logger.logger.info(f"Starting processing for article {article_id}")

        # Validate input structure
        error = self.validator.validate_article(article)
        if error:
            self.logger.logger.error(f"Validation failed for article {article_id}: {error}")
            return None, None

        try:
            # 1) Extract
            extract_result, error = self.process_extract_phase(article)
            if error or extract_result is None:
                return None, None

            # 2) Contextualize
            context_result, error = self.process_contextualize_phase(
                article,
                extract_result["statements"]
            )
            if error or context_result is None:
                return None, None

            # 3) Trim
            trim_result, error = self.process_trim_phase(
                article,
                extract_result["statements"],       # Original extracts
                context_result.get("context_added", [])
            )
            if error or trim_result is None:
                return None, None

            # 4) Merge
            merge_result, error = self.process_merge_phase(
                article,
                trim_result["statements"]
            )
            if error or merge_result is None:
                return None, None

            # 5) Temporal
            temporal_result, error = self.process_temporal_phase(
                article,
                merge_result["statements"]
            )
            if error or temporal_result is None:
                return None, None

            # 6) Remainder (NEW PHASE)
            remainder_result, error = self.process_remainder_phase(
                article,
                temporal_result["statements"]
            )
            if error or remainder_result is None:
                return None, None

            # 7) Mechanism
            mechanism_result, error = self.process_exclude_phases(
                article,
                remainder_result["statements"]
            )
            if error or mechanism_result is None:
                return None, None

            # 8) Validate
            validate_result, error = self.process_validate_phase(
                article,
                mechanism_result["mechanism"]["mechanism_statements"],
                mechanism_result["mechanism"]["blanket_statements"]
            )
            if error or validate_result is None:
                return None, None

            ##################################################################
            # BUILD FINAL OUTPUT
            ##################################################################
            extracted_flat = self._transform_individual_statements(
                extract_result.get("individual_statements", [])
            )
            contextualize_side_by_side = self._pair_contextualize_output(
                extract_result, context_result
            )
            trim_flat = self._transform_individual_statements(
                trim_result.get("individual_statements", [])
            )
            merge_flat = self._transform_individual_statements(
                merge_result.get("individual_statements", [])
            )
            temporal_flat = self._transform_individual_statements(
                temporal_result.get("individual_statements", [])
            )
            remainder_flat = {}
            if remainder_result:
                remainder_flat = self._transform_individual_statements(
                    remainder_result.get("individual_statements", [])
                )
            mechanism_flat = self._transform_individual_statements(
                mechanism_result["mechanism"].get("individual_statements", [])
            )
            validate_flat = self._transform_individual_statements(
                validate_result.get("individual_statements", [])
            )

            # Collect final discards from validate, if any
            discard_info = None
            if validate_result.get("mechanism_discarded") or validate_result.get("blanket_discarded"):
                discard_info = {
                    "id": article_id,
                    "fulltext": article["fulltext"],
                    "discarded_statements": {
                        "mechanism": validate_result.get("mechanism_discarded", []),
                        "blanket": validate_result.get("blanket_discarded", [])
                    }
                }

            final_data = {
                "id": article_id,
                "fulltext": article["fulltext"],
                "processing_stages": {
                    "extract": extracted_flat,
                    "contextualize": contextualize_side_by_side,
                    "trim": trim_flat,
                    "merge": merge_flat,
                    "temporal": temporal_flat,
                    "remainder": remainder_flat,
                    "mechanism": mechanism_flat,
                    "validation": validate_flat
                }
            }

            self.logger.logger.info(
                f"Successfully processed article {article_id}",
                extra={"article_id": article_id}
            )
            return final_data, discard_info

        except Exception as e:
            msg = f"Processing failed for article {article_id}: {str(e)}"
            self.logger.logger.error(msg, exc_info=True)
            return None, None
