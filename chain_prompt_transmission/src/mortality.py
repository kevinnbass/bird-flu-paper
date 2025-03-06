#!/usr/bin/env python3
"""
mortality.py

Usage:
  python3 -m src.mortality [INPUT_FILE] [CONFIG_FILE] [API_MODE] [--concurrency N] [--temperature T]

Key Changes:
 - Each phase (extract, context, nuance, nuance_filter) now retries up to 10 times on error/invalid response.
 - If all 10 attempts fail for a phase, we add an "errors" field to track failures, log all responses 
   to a special file (failed_articles_outputs.jsonl), but continue processing with partial data.
 - Failed phases now set "mortality=error" or "nuance=error" rather than defaulting to "no", allowing
   for clear distinction between genuine lack of content versus processing failures.
 - In case of nuance_filter failure, we fall back to using unfiltered analysis data.
 - We maintain the "pipeline_failure" field for backward compatibility.
 - We still record logs in "issues" as before.
 - TROUBLESHOOT_MODE is now True => we only process the specified IDs in TROUBLE_IDS_RAW.
 - We also rely on the updated openrouter_api_2.py for rate-limit (1 call/sec).
 - Added a final nuance-filter phase that takes the raw nuance analysis
   and filters out any items that are not "real" nuances.
 - Fixed KeyError by escaping braces in MORTALITY_EXTRACT_PROMPT (double braces).
 - Real-time output: Results are written to the output file as they are processed.
   Each article's result is flushed to disk immediately after processing to allow monitoring
   the output file while the script is running.
"""

import os
import sys
import re
import json
import yaml
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add the current directory to the path to ensure imports work correctly
sys.path.append('.')
sys.path.append('./src')

# The openrouter_api_2.py now has a rate-limit so we never start more than
# one new API call within the same second.
from openrouter_api_2 import OpenRouterClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("processing_mortality_nuance.log")
file_handler.setLevel(logging.INFO)
file_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_fmt)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_fmt)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

##############################################################################
#                         Troubleshoot Mode (Now True)                       #
##############################################################################

TROUBLESHOOT_MODE = True

# Replaced original IDs with the ones you provided
TROUBLE_IDS_RAW = [
    291, 1483, 421, 1728, 570, 1990, 940, 2002, 1894, 1326, 1434, 1254, 1146, 251,
    619, 1580, 451, 589, 260, 1653, 1922, 2003, 1322, 1978, 670, 1165, 468,
    278, 1712, 671, 406, 1478, 1887, 1242, 1470, 2000, 748, 812, 1497, 2006, 926,
    1096, 1021, 941, 674, 1698, 532, 569, 2025, 1538, 1780, 738, 1109, 400, 567,
    1506, 418, 2008, 265, 1435, 1993, 756, 62, 1925, 1451, 33, 338, 271, 1621,
    1446, 1988, 1706, 1878, 502, 452, 147, 1482, 811, 2009, 1440, 1504, 2011, 1716,
    1496, 494, 1598, 529, 1895, 1293, 1334, 647, 1924, 997, 453, 1568, 2023, 1841,
    573, 636,
    # Additional nuance='no' => nuance='yes' test IDs:
    151, 1643, 1472, 258, 150, 1223, 1794,
    27, 20
]
TROUBLE_IDS = set(map(str, TROUBLE_IDS_RAW))

##############################################################################
#                   Helper: get_next_output_filename                         #
##############################################################################

def get_next_output_filename(dir_path: Path) -> Path:
    """
    Find the next available output filename by looking at existing files
    and incrementing the highest number found, including troubleshoot files.
    """
    # Check for both normal and troubleshoot files to find the highest number
    max_num = 0
    
    # Pattern for regular files
    regular_pattern = re.compile(r"^mortality_nuance_(\d+)\.jsonl$")
    for item in dir_path.glob("mortality_nuance_*.jsonl"):
        match = regular_pattern.match(item.name)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    
    # Pattern for troubleshoot files
    troubleshoot_pattern = re.compile(r"^mortality_nuance_(\d+)_troubleshoot\.jsonl$")
    for item in dir_path.glob("mortality_nuance_*_troubleshoot.jsonl"):
        match = troubleshoot_pattern.match(item.name)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    
    # Return the next filename (with base name only - troubleshoot suffix will be added later if needed)
    return dir_path / f"mortality_nuance_{max_num+1}.jsonl"

##############################################################################
#                           Chain Prompts (Escaped)                          #
##############################################################################

# IMPORTANT: Double-brace the literal braces
MORTALITY_EXTRACT_PROMPT = """\
You are an analytical assistant.

Identify statements that mention a high HUMAN mortality rate or that the virus is deadly.

Return those statements from the text verbatim.

Respond ONLY with a JSON object in the format:
{{
  "mortality_statements": []
}}
If none found, keep it empty.

Text:
{fulltext}

IMPORTANT OUTPUT REQUIREMENTS:
- Do not include explanations, disclaimers, or any extra text.
- If no statements are found, return an empty array.
- Escape any quotes inside statements.
- Always return complete sentences, not partial sentences.
"""

MORTALITY_CONTEXTUALIZE_PROMPT = """\
You are an assistant. DO NOT REVEAL YOUR CHAIN OF THOUGHT.

For each statement below, extract exactly 3 sentences before and after, verbatim. This is a PURELY MECHANICAL EXTRACTION TASK.

FOLLOW THIS ALGORITHM EXACTLY - DO NOT DEVIATE:

STEP 1: Create a "contexts" array to hold your results
STEP 2: For each mortality statement in the provided list:
  STEP 2.1: Locate the EXACT statement in the article text
  STEP 2.2: Mark the exact position where this statement starts and ends
  STEP 2.3: Starting from the start position, move BACKWARD to collect up to 3 sentences
  STEP 2.4: Starting from the end position, move FORWARD to collect up to 3 sentences
  STEP 2.5: Format these as {{"before": [...], "statement": "...", "after": [...]}}
  STEP 2.6: Add this object to your "contexts" array
STEP 3: Return ONLY the JSON with the "contexts" array

SENTENCE TRAVERSAL RULES:
- A new sentence begins IMMEDIATELY after a period, question mark, or exclamation point followed by a space
- Collect sentences in the order you encounter them
- If you reach the beginning/end of the text before finding 3 sentences, stop
- Skip any promotional content, navigation elements, or advertisements
- DO NOT count or label sentences (A, B, C, etc.) - just extract them in order

SENTENCE COLLECTION PROCEDURE:
- BACKWARD collection: Start immediately before the statement; gather up to 3 valid sentences moving backward
- FORWARD collection: Start immediately after the statement; gather up to 3 valid sentences moving forward
- When the mortality statement contains multiple sentences, count from the beginning of the first sentence and after the end of the last sentence

IRRELEVANT CONTENT RULES:
- TEXT CONTAINING "Related:", "UPCOMING EVENT", "Privacy Policy", etc. = IGNORE COMPLETELY
- ADVERTISEMENTS, SIGNUP FORMS, NAVIGATION MENUS = IGNORE COMPLETELY
- When you encounter irrelevant content while collecting, SKIP IT entirely and continue collecting from valid content

JSON FORMATTING:
- Properly escape ALL quotes with backslash: \\"
- Ensure all strings have both opening and closing quotes
- Double-check that each string is properly terminated

Example JSON structure:
{{
  "contexts": [
    {{
      "before": ["First sentence before.", "Second sentence before.", "Third sentence before."],
      "statement": "This is the mortality statement that was found exactly as written.",
      "after": ["First sentence after.", "Second sentence after.", "Third sentence after."]
    }}
  ]
}}

Article Text:
{fulltext}

Mortality Statements:
{statements_list}

CRITICAL REQUIREMENTS:
- Return ONLY valid JSON - nothing else
- No explanations, comments, or text outside the JSON
- No code blocks or backticks
- EXECUTE THE ALGORITHM MECHANICALLY WITHOUT THINKING ABOUT MEANING
- DO NOT try to understand the content - just extract text based on position
- If you find yourself doing complex reasoning, STOP and follow the mechanical procedure
"""

MORTALITY_NUANCE_PROMPT = """\
You must output ONLY a JSON object. 
DO NOT PROVIDE ANY REASONING OR EXPLANATION.
NO REASONING. ONLY JSON OUTPUT.

Determine if this text provides any nuance that calls into question the accuracy of a reported high mortality rate.

EXTRACT ONLY VERBATIM statements from the text that provide this nuance. DO NOT generate your own interpretations.

Your output MUST be a valid parseable JSON object with this exact format and nothing else:
{{
  "analysis": [
    {{
      "statement": "exact statement from context",
      "nuances": ["exact verbatim text 1", "exact verbatim text 2"]
    }}
  ]
}}

CRUCIAL INSTRUCTION: Both the "statement" and each item in the "nuances" array MUST be DIRECT QUOTES copied word-for-word from the provided context. Do not paraphrase, summarize, or create new content.

Contexts:
{contexts_json}

CRITICAL OUTPUT REQUIREMENTS:
- Return ONLY a valid JSON object exactly as shown above - nothing else
- No reasoning, no explanations, no comments
- No phrases like "here is the JSON" or "the JSON output is" 
- No backticks, no markdown, no code blocks
- Identify statements expressing doubt or nuance about the mortality rate
- The array should contain objects with "statement" and "nuances" keys
- For "nuances", extract ONLY direct quotes from the text that provide nuance about mortality rates
- NEVER generate your own analysis or interpretations - use exact quotes only
- Escape any quotes inside statements
- Always return complete sentences, not partial sentences
"""

MORTALITY_NUANCE_FILTER_PROMPT = """\
You must output ONLY a JSON object.
DO NOT PROVIDE ANY REASONING OR EXPLANATION.
NO REASONING. ONLY JSON OUTPUT.

Filter the following analysis to identify only statements that genuinely question or clarify mortality rate accuracy.

Your output MUST be a valid parseable JSON object with this exact format and nothing else:
{{
  "filtered_analysis": [
    {{
      "statement": "exact statement from original analysis",
      "nuances": ["exact verbatim text 1", "exact verbatim text 2"]
    }}
  ]
}}

CRUCIAL INSTRUCTION: Both the "statement" and each item in the "nuances" array MUST be DIRECT QUOTES copied word-for-word from the provided analysis. Do not paraphrase, summarize, or create new content.

NOT considered nuances:
- general facts about the disease
- statements about transmission mode
- symptoms description
- the infection can be mild in some cases
- some cases aren't being detected

Some examples include:

Example 1: "The strain of the virus circulating in many parts of the world including the United States seems to trigger human infections less frequently than earlier versions of the virus did."
Example 2: "The disease in humans ranges from mild infections, which include upper-respiratory and eye-related symptoms, to severe pneumonia."
Example 3: "We had no evidence these were person-to-person transmitted cases."
Example 4: "There has been not enough testing of people who had been contact with infected birds."
Example 5: "Experts caution that the risk to the general public is low and that there is no evidence of sustained human-to-human transmission."
Example 6: "Some people don't have noticeable symptoms."
Example 7: "The H5N1 virus has rarely infected humans."
Example 8: "Although the three farmworkers diagnosed with the disease this year in the United States had only mild symptoms, like a runny nose and inflamed eyes, others may not be so lucky."
Example 9: "In the past, H5N1 has led to severe illness in humans, with a global fatality rate of more than 50 percent, but these cases were mild."

Analysis:
{analysis_json}

CRITICAL OUTPUT REQUIREMENTS:
- Return ONLY a valid JSON object exactly as shown above - nothing else
- No reasoning, no explanations, no comments
- No phrases like "here is the JSON" or "the JSON output is" 
- No backticks, no markdown, no code blocks
- Filter out items that do NOT actually question or clarify mortality rate accuracy
- The array objects must have "statement" and "nuances" keys
- ONLY use direct verbatim quotes from the text - never generate interpretations
- NEVER create your own analysis, summaries, or paraphrasing
- Escape any quotes inside statements
"""

##############################################################################
#                   MortalityNuanceProcessor (Async) with 10 Retries        #
##############################################################################

@dataclass
class MortalityNuanceProcessor:
    api_client: OpenRouterClient
    
    def log_failed_article(self, article_id: str, phase: str, all_responses: List[Any]) -> None:
        """
        Log all outputs from a failed article processing to a special file.
        """
        failed_articles_log = Path("failed_articles_outputs.jsonl")
        try:
            output = {
                "id": article_id,
                "phase": phase,
                "timestamp": self._get_timestamp(),
                "responses": all_responses
            }
            with open(failed_articles_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
                f.flush()
            logger.info(f"Logged all outputs for failed article {article_id} in phase {phase} to {failed_articles_log}")
        except Exception as e:
            logger.error(f"Failed to log outputs for article {article_id}: {e}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()

    async def mortality_extract_phase(self, article: Dict[str, Any], issues: List[str]) -> Optional[List[str]]:
        """
        Up to 10 attempts. If all fail => return None => pipeline_failure='extract'.
        """
        article_id = str(article.get("id", ""))
        fulltext = article.get("fulltext", "")
        prompt = MORTALITY_EXTRACT_PROMPT.format(fulltext=fulltext)

        max_tries = 10
        all_responses = []
        
        for attempt in range(1, max_tries + 1):
            logger.info(f"[EXTRACT] Attempt {attempt}/{max_tries} for article {article_id}...")
            # Pass fulltext as context data for potential backup usage
            context_data = {"fulltext": fulltext}
            response = await self.api_client.make_call(prompt, article_id, phase="extract", context_data=context_data)
            # Store all responses for logging if all attempts fail
            all_responses.append(response)
            
            if isinstance(response, dict) and "error" in response:
                msg = f"EXTRACT API error for article {article_id}: {response['error']}"
                logger.error(msg)
                issues.append(msg)
                if attempt < max_tries:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.log_failed_article(article_id, "extract", all_responses)
                    return None
            if not isinstance(response, dict):
                msg = f"EXTRACT got non-dict response for {article_id}."
                logger.warning(msg)
                issues.append(msg)
                if attempt < max_tries:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.log_failed_article(article_id, "extract", all_responses)
                    return None

            statements = response.get("mortality_statements", [])
            if not isinstance(statements, list):
                msg = f"EXTRACT: 'mortality_statements' is not a list for article {article_id}."
                logger.warning(msg)
                issues.append(msg)
                if attempt < max_tries:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.log_failed_article(article_id, "extract", all_responses)
                    return None

            if any(not isinstance(s, str) for s in statements):
                msg = f"EXTRACT: One or more statements are non-string for article {article_id}."
                logger.warning(msg)
                issues.append(msg)
                if attempt < max_tries:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.log_failed_article(article_id, "extract", all_responses)
                    return None

            logger.info(f"[EXTRACT] Found {len(statements)} valid mortality statements for article {article_id}.")
            return statements

        self.log_failed_article(article_id, "extract", all_responses)
        return None

    async def mortality_contextualize_phase(
        self,
        article: Dict[str, Any],
        statements: List[str],
        issues: List[str]
    ) -> Optional[List[Dict[str, str]]]:
        """
        Up to 10 attempts. If all fail => return None => pipeline_failure='context'.
        """
        article_id = str(article.get("id", ""))
        if not statements:
            return []

        fulltext = article.get("fulltext", "")
        prompt = MORTALITY_CONTEXTUALIZE_PROMPT.format(
            fulltext=fulltext,
            statements_list=json.dumps(statements, ensure_ascii=False)
        )

        max_tries = 10
        all_responses = []
        
        for attempt in range(1, max_tries + 1):
            logger.info(f"[CONTEXT] Attempt {attempt}/{max_tries} for article {article_id}...")
            # Pass fulltext as context data for potential backup usage
            context_data = {"fulltext": fulltext}
            response = await self.api_client.make_call(prompt, article_id, phase="context", context_data=context_data)
            # Store all responses for logging if all attempts fail
            all_responses.append(response)
            
            if isinstance(response, dict) and "error" in response:
                msg = f"CONTEXT API error for article {article_id}: {response['error']}"
                logger.error(msg)
                issues.append(msg)
                if attempt < max_tries:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.log_failed_article(article_id, "context", all_responses)
                    return None
            if not isinstance(response, dict):
                msg = f"CONTEXT got non-dict response for {article_id}."
                logger.warning(msg)
                issues.append(msg)
                if attempt < max_tries:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.log_failed_article(article_id, "context", all_responses)
                    return None

            contexts = response.get("contexts", [])
            if not isinstance(contexts, list):
                msg = f"CONTEXT: 'contexts' is not a list for article {article_id}."
                logger.warning(msg)
                issues.append(msg)
                if attempt < max_tries:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.log_failed_article(article_id, "context", all_responses)
                    return None

            logger.info(f"[CONTEXT] Received {len(contexts)} context items for article {article_id}.")
            return contexts

        self.log_failed_article(article_id, "context", all_responses)
        return None

    async def mortality_nuance_phase(
        self,
        article: Dict[str, Any],
        contexts: List[Dict[str, str]],
        issues: List[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Up to 10 attempts. If all fail => return None => pipeline_failure='nuance'.
        """
        article_id = str(article.get("id", ""))
        if not contexts:
            return []

        fulltext = article.get("fulltext", "")
        contexts_json_str = json.dumps(contexts, ensure_ascii=False)
        prompt = MORTALITY_NUANCE_PROMPT.format(contexts_json=contexts_json_str)

        max_tries = 10
        all_responses = []
        
        for attempt in range(1, max_tries + 1):
            logger.info(f"[NUANCE] Attempt {attempt}/{max_tries} for article {article_id}...")
            # Pass fulltext as context data for potential backup usage
            context_data = {"fulltext": fulltext}
            response = await self.api_client.make_call(prompt, article_id, phase="nuance", context_data=context_data)
            # Store all responses for logging if all attempts fail
            all_responses.append(response)
            
            if isinstance(response, dict) and "error" in response:
                msg = f"NUANCE API error for article {article_id}: {response['error']}"
                logger.error(msg)
                issues.append(msg)
                if attempt < max_tries:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.log_failed_article(article_id, "nuance", all_responses)
                    return None
            if not isinstance(response, dict):
                msg = f"NUANCE got non-dict response for {article_id}."
                logger.warning(msg)
                issues.append(msg)
                if attempt < max_tries:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.log_failed_article(article_id, "nuance", all_responses)
                    return None

            analysis = response.get("analysis", [])
            if not isinstance(analysis, list):
                msg = f"NUANCE: 'analysis' is not a list for article {article_id}."
                logger.warning(msg)
                issues.append(msg)
                if attempt < max_tries:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.log_failed_article(article_id, "nuance", all_responses)
                    return None

            if not all(isinstance(a, dict) for a in analysis):
                msg = f"NUANCE analysis for article {article_id} has non-dict items."
                logger.warning(msg)
                issues.append(msg)
                if attempt < max_tries:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.log_failed_article(article_id, "nuance", all_responses)
                    return None

            logger.info(f"[NUANCE] Received {len(analysis)} analyses for article {article_id}.")
            return analysis

        self.log_failed_article(article_id, "nuance", all_responses)
        return None

    async def mortality_nuance_filter_phase(
        self,
        article: Dict[str, Any],
        raw_analysis: List[Dict[str, Any]],
        issues: List[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Up to 10 attempts. If all fail => return None => pipeline_failure='nuance_filter'.
        This step filters out any 'analysis' items that are not real nuances.
        """
        article_id = str(article.get("id", ""))
        if not raw_analysis:
            return []

        fulltext = article.get("fulltext", "")
        analysis_json_str = json.dumps(raw_analysis, ensure_ascii=False)
        prompt = MORTALITY_NUANCE_FILTER_PROMPT.format(analysis_json=analysis_json_str)

        max_tries = 10
        all_responses = []
        
        for attempt in range(1, max_tries + 1):
            logger.info(f"[NUANCE_FILTER] Attempt {attempt}/{max_tries} for article {article_id}...")
            # Pass fulltext as context data for potential backup usage
            context_data = {"fulltext": fulltext}
            response = await self.api_client.make_call(prompt, article_id, phase="nuance_filter", context_data=context_data)
            # Store all responses for logging if all attempts fail
            all_responses.append(response)
            
            if isinstance(response, dict) and "error" in response:
                msg = f"NUANCE_FILTER API error for article {article_id}: {response['error']}"
                logger.error(msg)
                issues.append(msg)
                if attempt < max_tries:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.log_failed_article(article_id, "nuance_filter", all_responses)
                    return None
            if not isinstance(response, dict):
                msg = f"NUANCE_FILTER got non-dict response for {article_id}."
                logger.warning(msg)
                issues.append(msg)
                if attempt < max_tries:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.log_failed_article(article_id, "nuance_filter", all_responses)
                    return None

            filtered = response.get("filtered_analysis", [])
            if not isinstance(filtered, list):
                msg = f"NUANCE_FILTER: 'filtered_analysis' is not a list for article {article_id}."
                logger.warning(msg)
                issues.append(msg)
                if attempt < max_tries:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.log_failed_article(article_id, "nuance_filter", all_responses)
                    return None

            if not all(isinstance(a, dict) for a in filtered):
                msg = f"NUANCE_FILTER: filtered_analysis has non-dict items for {article_id}."
                logger.warning(msg)
                issues.append(msg)
                if attempt < max_tries:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.log_failed_article(article_id, "nuance_filter", all_responses)
                    return None

            logger.info(f"[NUANCE_FILTER] Received {len(filtered)} filtered analyses for article {article_id}.")
            return filtered

        self.log_failed_article(article_id, "nuance_filter", all_responses)
        return None

    async def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pipeline with four phases: extract -> context -> nuance -> nuance_filter.
        Processing continues even if a phase fails, adding appropriate error indicators.
        """
        article_id = str(article.get("id", ""))
        logger.info(f"Processing article {article_id}...")

        issues: List[str] = []
        errors: Dict[str, str] = {}  # Track errors by phase

        date = article.get("date", "")
        title = article.get("title", "")
        outlet = article.get("media_outlet", "")
        fulltext = article.get("fulltext", "")

        # Initialize the final record with base info
        final_rec = {
            "id": article_id,
            "date": date,
            "fulltext": fulltext,
            "media_outlet": outlet,
            "title": title,
        }

        # 1) EXTRACT
        extracted = await self.mortality_extract_phase(article, issues)
        if extracted is None:
            errors["extract"] = "All retries failed"
            logger.error(f"[ARTICLE] All retries failed in extract for {article_id}. Setting mortality=error.")
            final_rec["mortality"] = "error"
            final_rec["errors"] = errors
            if issues:
                final_rec["issues"] = issues
            # Don't set nuance to error, leave it undetermined
            final_rec["nuance"] = "undetermined"
            return final_rec

        if not extracted:
            # no statements => mortality=no
            logger.info(f"[ARTICLE] No mortality statements => mortality=no. {article_id}")
            final_rec["mortality"] = "no"
            if issues:
                final_rec["issues"] = issues
            # Don't set nuance to error or no, mark it as not applicable when no mortality statements
            final_rec["nuance"] = "not_applicable"
            return final_rec

        # Add mortality statements to the record
        final_rec["mortality"] = "yes"
        for idx, st_text in enumerate(extracted, start=1):
            if not isinstance(st_text, str):
                msg = f"EXTRACTed statement not a string => ignoring. article {article_id}"
                logger.warning(msg)
                issues.append(msg)
                continue
            m_label = f"mortality_{idx}"
            final_rec[m_label] = st_text

        # 2) CONTEXT
        contexts = await self.mortality_contextualize_phase(article, extracted, issues)
        if contexts is None:
            errors["context"] = "All retries failed"
            logger.error(f"[ARTICLE] All retries failed in context for {article_id}. Continuing with partial data.")
            final_rec["errors"] = errors
            if issues:
                final_rec["issues"] = issues
            # Don't set nuance to error, mark it as undetermined due to context phase failure
            final_rec["nuance"] = "undetermined"
            return final_rec

        # 3) NUANCE
        analysis = await self.mortality_nuance_phase(article, contexts, issues)
        if analysis is None:
            errors["nuance"] = "All retries failed"
            logger.error(f"[ARTICLE] All retries failed in nuance for {article_id}. Continuing with partial data.")
            final_rec["errors"] = errors
            if issues:
                final_rec["issues"] = issues
            # Don't set nuance to error, mark it as undetermined due to nuance phase failure
            final_rec["nuance"] = "undetermined"
            return final_rec

        # 4) NUANCE_FILTER
        filtered_analysis = await self.mortality_nuance_filter_phase(article, analysis, issues)
        if filtered_analysis is None:
            errors["nuance_filter"] = "All retries failed"
            logger.error(f"[ARTICLE] All retries failed in nuance_filter for {article_id}. Using unfiltered analysis.")
            
            # Use the unfiltered analysis data since filtering failed
            filtered_analysis = analysis
            final_rec["errors"] = errors
        
        # Process nuances from the analysis (filtered or unfiltered)
        nuance_found = False
        statement_map = {}

        # Map statements to their nuances
        for a in filtered_analysis:
            st = a.get("statement", None)
            if not isinstance(st, str):
                msg = f"Analysis item missing 'statement' string => ignoring item for {article_id}"
                logger.warning(msg)
                issues.append(msg)
                continue
            # "nuances" might be a list of strings
            statement_map[st] = a.get("nuances", [])

        # Add nuances to the final record
        for idx, st_text in enumerate(extracted, start=1):
            if not isinstance(st_text, str):
                continue  # Already logged in the mortality statements loop
            
            these_nuances = statement_map.get(st_text, [])
            if these_nuances:
                nuance_found = True
                for n_idx, snippet in enumerate(these_nuances, start=1):
                    # nuance_1, nuance_1a, nuance_1b, ...
                    nuance_label = f"nuance_{idx}" if n_idx == 1 else f"nuance_{idx}{chr(96 + n_idx)}"
                    final_rec[nuance_label] = snippet

        # Only set nuance to "no" if we definitely found no nuances after a successful analysis
        # Otherwise, if there were errors in the analysis, mark as "possible"
        if nuance_found:
            final_rec["nuance"] = "yes"
        elif errors:
            # If there were errors, don't default to "no" - use "possible" instead
            final_rec["nuance"] = "possible"
        else:
            final_rec["nuance"] = "no"
        
        if issues:
            final_rec["issues"] = issues
            
        # For backward compatibility
        if errors:
            # Set pipeline_failure to first error phase for compatibility
            final_rec["pipeline_failure"] = next(iter(errors.keys()))
            
        logger.info(
            f"[ARTICLE] Done {article_id} => mortality={final_rec['mortality']} nuance={final_rec['nuance']} errors={errors}"
        )
        return final_rec

##############################################################################
#                           Final Sync Step                                  #
##############################################################################

def read_reference_file(ref_path: str) -> Dict[str, Dict[str, Any]]:
    ref_map = {}
    try:
        with open(ref_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                rid = str(data.get("id", ""))
                if rid:
                    ref_map[rid] = data
    except Exception as e:
        logger.error(f"Error reading reference file {ref_path}: {e}")
    return ref_map

def sync_with_reference(
    output_records: List[Dict[str, Any]],
    reference_map: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    for out_rec in output_records:
        out_id = str(out_rec.get("id", ""))
        if out_id in reference_map:
            ref_rec = reference_map[out_id]
            if "mortality" in ref_rec:
                out_rec["mortality"] = ref_rec["mortality"]
            if "nuance" in ref_rec:
                out_rec["nuance"] = ref_rec["nuance"]
            for k, v in ref_rec.items():
                v_str = v if isinstance(v, str) else ""
                if ("miscoded" in k.lower()) or ("miscoded" in v_str.lower()):
                    out_rec[k] = v
    return output_records

##############################################################################
#                          Main (CLI + Async Runner)                         #
##############################################################################

async def async_main(input_file: str, config_file: str, api_mode: str, concurrency: int, temperature_override: Optional[float] = None):
    """
    Process all articles in async mode for maximum throughput.
    Uses asyncio.Queue for bounded concurrency.
    """
    # Setup the API client
    logger.info(f"Loading config from {config_file}...")
    with open(config_file, "r", encoding="utf-8") as cf:
        raw_conf = yaml.safe_load(cf)

    if "api" not in raw_conf or "openrouter" not in raw_conf["api"]:
        logger.error("config.yaml missing 'api.openrouter' block. Cannot proceed.")
        return

    # Apply temperature override if provided
    if temperature_override is not None:
        logger.info(f"Overriding temperature with: {temperature_override}")
        raw_conf["api"]["openrouter"]["temperature"] = temperature_override

    # Initialize OpenRouter client
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not openrouter_api_key:
        logger.error("No OPENROUTER_API_KEY found in environment. Stopping.")
        return

    # Initialize OpenAI client as backup
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        logger.warning("No OPENAI_API_KEY found in environment. Will proceed without OpenAI backup.")
        openai_client = None
    else:
        from openai_api import APIClient
        from pathlib import Path
        logger.info("Initializing OpenAI backup client with o3-mini model...")
        try:
            openai_client = APIClient(openai_api_key, Path(config_file))
            # Match settings to ensure identical outputs
            if temperature_override is not None:
                openai_client.config.temperature = temperature_override
            
            logger.info(f"OpenAI backup client initialized with model {raw_conf['api']['openai']['model']}")
            
            # Set the debug mode for the OpenAI client to match the OpenRouter client's debug setting
            # This ensures debugging logs are consistent
            if "debug_mode" in raw_conf["api"]["openrouter"]:
                openai_client.debug_mode = raw_conf["api"]["openrouter"]["debug_mode"]
                logger.info(f"Setting OpenAI client debug mode to {openai_client.debug_mode} to match OpenRouter client")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI backup client: {e}")
            openai_client = None

    # Initialize the OpenRouter client
    client = OpenRouterClient(openrouter_api_key, raw_conf["api"]["openrouter"], temperature_override)
    
    # Set up the OpenAI backup if available
    if openai_client:
        client.setup_openai_backup(openai_client)
        logger.info("OpenAI backup configured for OpenRouter client")
    
    logger.info(f"Testing OpenRouter connection to model '{client.model}' with temperature {client.temperature}...")
    try:
        await client.test_connection()
        logger.info("OpenRouter connection test passed.")
    except Exception as exc:
        logger.error(f"Connection test failed: {exc}")
        return

    processor = MortalityNuanceProcessor(api_client=client)

    in_path = Path(input_file)
    logger.info(f"Reading articles from {in_path}...")
    articles = []
    try:
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    articles.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Cannot read input file {input_file}: {e}")
        return
    logger.info(f"Found {len(articles)} articles in {input_file}.")

    out_dir = in_path.parent
    output_path = get_next_output_filename(out_dir)

    # Because TROUBLESHOOT_MODE is True, rename with "_troubleshoot"
    if TROUBLESHOOT_MODE:
        output_path = output_path.with_name(output_path.stem + "_troubleshoot.jsonl")

    logger.info(f"This run's output file will be: {output_path.name}")

    if TROUBLESHOOT_MODE:
        filtered = []
        for art in articles:
            aid_str = str(art.get("id", ""))
            if aid_str in TROUBLE_IDS:
                filtered.append(art)
        articles = filtered
        logger.info(f"TROUBLESHOOT_MODE on. Only {len(articles)} articles remain after filtering trouble IDs.")

    logger.info(f"Beginning chain processing with concurrency={concurrency}...")

    semaphore = asyncio.Semaphore(concurrency)
    results_dict: Dict[int, Dict[str, Any]] = {}
    next_to_write = 0
    lock = asyncio.Lock()

    async def worker(idx: int, art: Dict[str, Any]):
        async with semaphore:
            return (idx, await processor.process_article(art))

    tasks = []
    for i, art in enumerate(articles):
        tasks.append(asyncio.create_task(worker(i, art)))

    with open(output_path, "w", encoding="utf-8") as out_f:
        for fut in asyncio.as_completed(tasks):
            idx, rec = await fut
            async with lock:
                results_dict[idx] = rec
                while next_to_write in results_dict:
                    out_f.write(json.dumps(results_dict[next_to_write], ensure_ascii=False) + "\n")
                    out_f.flush()  # Ensure output is written in real-time
                    del results_dict[next_to_write]
                    next_to_write += 1
                    logger.info(f"Processed article {next_to_write-1} of {len(articles)} - written to output in real-time")

    logger.info(f"Done processing. Wrote {next_to_write} articles to {output_path}.")

    # Final step: sync with reference
    reference_file = "/home/kevinnbass/big-data/github_bird_flu/chain_prompt_transmission/inputs/mortality_final_nuance_shorn_no_dups.jsonl"
    logger.info(f"Reading reference file: {reference_file}")
    ref_map = read_reference_file(reference_file)

    logger.info("Syncing newly produced output with reference data...")
    final_records = []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    final_records.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to re-read output file {output_path} for sync step: {e}")
        return

    synced_records = sync_with_reference(final_records, ref_map)

    final_output_path = output_path.with_name(output_path.stem + "_final.jsonl")
    logger.info(f"Writing final synced output to {final_output_path}")
    with open(final_output_path, "w", encoding="utf-8") as out_file:
        for i, record in enumerate(synced_records):
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()  # Ensure synced output is written in real-time
            if i % 10 == 0:  # Log progress every 10 records
                logger.info(f"Synced and wrote {i+1}/{len(synced_records)} records in real-time")

    logger.info("Final sync completed.")

def main():
    parser = argparse.ArgumentParser(
        description="Mortality chain with concurrency, 10 retries, pipeline_failure on phase fail, logs failed outputs, braces escaped, normal mode."
    )
    parser.add_argument("input_file", help="Path to the input JSONL file.")
    parser.add_argument("config_file", help="Path to config.yaml.")
    parser.add_argument("api_mode", nargs="?", default="deepseek", help="Unused except logs.")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent requests.")
    parser.add_argument("--temperature", type=float, help="Override the temperature setting in the config.")
    args = parser.parse_args()

    temperature_info = f", temperature={args.temperature}" if args.temperature is not None else ""
    logger.info(
        f"Starting mortality chain. input_file={args.input_file}, config_file={args.config_file}, "
        f"api_mode={args.api_mode}, concurrency={args.concurrency}{temperature_info}"
    )
    asyncio.run(async_main(args.input_file, args.config_file, args.api_mode, args.concurrency, args.temperature))

if __name__ == "__main__":
    main()
