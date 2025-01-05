#!/usr/bin/env python3
"""
Integration script that:

Pre-Pipeline (Match & Transfer):
--------------------------------
A) Reads processed_all_articles_examine_complete_with_huffpost.json (the "examine" file)
B) Reads processed_all_articles_fixed_5.jsonl (the "5-file")
C) Matches entries by:
   1) (title, media_outlet) for certain "special" articles (ignoring fulltext),
   2) (title, media_outlet, fulltext=clean_fulltext) for all others
D) Transfers high_rate_2, nuance_2, comments, feedback_2 to the matching "5-file" entries,
   overwriting if already present
E) Prints how many entries matched AND which entries had NO match

Main Pipeline (transformations):
--------------------------------
1) Copy processed_all_articles_fixed_5.jsonl -> distribution_set.jsonl
2) Overwrite date from processed_all_articles_fixed_3.jsonl
3) Clean dash-like characters in clean_fulltext (replace them with a space)
4) Remove extra spaces
5) Remove space before punctuation
6) Rename clean_fulltext -> fulltext
7) Place id, title, date, fulltext in that order, then everything else
8) Remove unwanted fields (but keep high_rate_2, nuance_2, feedback_2, etc.)
   - Also remove 'feedback_sa' and all '_intensity' fields.

Additional Requirement:
-----------------------
- Also remove double/extra white spaces (and spaces before punctuation)
  from the 'title' field, just like we do for 'fulltext'.
- "id" field should be the very first field, then "title", then "date", then "fulltext".

Final Step:
-----------
- Create a file called distribution_mortality_subset.jsonl containing only
  articles with "high_rate_2" == "yes".

Run with:
    python3 pipeline.py
"""

import json
import re
import shutil
import os

# ----------------------------------------------------------------------
#  DEFINE SPECIAL MATCHING ARTICLES
# ----------------------------------------------------------------------
SPECIAL_MATCHES = {
    (
      "These are the bird flu questions that influenza and animal scientists desperately want answered",
      "STAT"
    ),
    (
      "11-year-old Cambodian girl dies from bird flu in country's first known human infection since 2014",
      "FoxNews"
    )
}

def build_match_key(title: str, media_outlet: str, fulltext: str):
    """
    Return either:
      (title, media_outlet) if in SPECIAL_MATCHES
      (title, media_outlet, fulltext) otherwise
    """
    if (title, media_outlet) in SPECIAL_MATCHES:
        return (title, media_outlet)
    else:
        return (title, media_outlet, fulltext)


# ----------------------------------------------------------------------
#  REMOVAL + CLEANING CONSTANTS
# ----------------------------------------------------------------------

# 1) Fields/keys to KEEP even if they match certain patterns
KEEP_KEYS = {
    "high_rate_2",
    "nuance_2",
    "feedback_2",
}

# 2) Regex patterns for fields to REMOVE (unless explicitly in KEEP_KEYS)
REMOVE_PATTERNS = [
    # Existing removal patterns
    r"^quotation_\d+$",
    r"^(joy|sadness|anger|fear|surprise|disgust|trust|anticipation|negative_sentiment|positive_sentiment|feedback_sa)_\d+$",
    r".*_fulltext$",
    r"^flesch_kincaid_grade_global$",
    r"^gunning_fog_global$",
    r".*_fulltext_intensity$",
    r".*_title_intensity$",
    r".*_title$",
    r"^feedback_sa_title$",
    r"^feedback_sa_\d+$",
    r"^nuance$",
    r"^high_rate$",
    r"^feedback$",
    r"^interviewee_\d+$",

    # Pattern to remove anger_1_intensity, fear_2_intensity, etc.
    r"^(anger|anticipation|disgust|fear|joy|sadness|surprise|trust|negative_sentiment|positive_sentiment)_[0-9]+_intensity$",

    # Explicitly remove 'feedback_sa'
    r"^feedback_sa$",
]

# 3) Dash-like characters to replace with a space (only for fulltext)
DASH_LIKE_CHARS = [
    "\u2014",  # em dash
    "\u2013",  # en dash
    "\u2015",  # horizontal bar
    "\u2012",
    "\u2010"
    # ...add more if needed
]


# ----------------------------------------------------------------------
#  CLEANING FUNCTIONS
# ----------------------------------------------------------------------

def clean_dashes(text: str) -> str:
    """Replace known dash-like characters with a single space."""
    for char in DASH_LIKE_CHARS:
        text = text.replace(char, " ")
    return text

def remove_extra_spaces(text: str) -> str:
    """
    Remove multiple spaces by splitting on whitespace
    and then rejoining with a single space.
    """
    return " ".join(text.split())

def remove_space_before_punct(text: str) -> str:
    """
    Remove whitespace before punctuation (, : ; . ! ?)
    e.g. "Hello , world !" => "Hello,world!"
    """
    return re.sub(r"\s+([,;:\.\?\!])", r"\1", text)

def remove_unwanted_fields(obj: dict) -> dict:
    """
    Removes fields from 'obj' that match any pattern in REMOVE_PATTERNS,
    unless they're explicitly in KEEP_KEYS.
    """
    cleaned = {}
    for k, v in obj.items():
        if k in KEEP_KEYS:
            # Always keep these keys
            cleaned[k] = v
            continue

        # Check if this key matches any remove pattern
        should_remove = any(re.match(pattern, k) for pattern in REMOVE_PATTERNS)
        if not should_remove:
            cleaned[k] = v

    return cleaned

def reorder_fields(obj: dict) -> dict:
    """
    Rebuild 'obj' so that it has this order:
       1) 'id'
       2) 'title'
       3) 'date'
       4) 'fulltext'
       5) all other fields (in original order).
    """
    new_obj = {}

    # 1. 'id' if present
    if "id" in obj:
        new_obj["id"] = obj["id"]

    # 2. 'title'
    if "title" in obj:
        new_obj["title"] = obj["title"]

    # 3. 'date'
    if "date" in obj:
        new_obj["date"] = obj["date"]

    # 4. 'fulltext'
    if "fulltext" in obj:
        new_obj["fulltext"] = obj["fulltext"]

    # 5. Add all other fields in the order they appear, skipping the above keys
    for k, v in obj.items():
        if k not in ("id", "title", "date", "fulltext"):
            new_obj[k] = v

    return new_obj


# ----------------------------------------------------------------------
#  MAIN PIPELINE
# ----------------------------------------------------------------------

def main():
    # Input/Output Filenames
    examine_file_json = "processed_all_articles_examine_complete_with_huffpost.json"
    file_5_jsonl      = "processed_all_articles_fixed_5.jsonl"
    file_3_jsonl      = "processed_all_articles_fixed_3.jsonl"
    dist_file         = "distribution_set.jsonl"
    temp_output_file  = "distribution_set_temp.jsonl"
    mortality_subset_file = "distribution_mortality_subset.jsonl"

    # --- Pre-Pipeline Step ---------------------------------------------
    # A) Read the examine file
    print(f"Reading examine file: {examine_file_json}")
    with open(examine_file_json, "r", encoding="utf-8") as f:
        examine_data = json.load(f)  # Could be a list or a dict with a list

    # We'll store them for fast lookup by (title, media_outlet) or (title, media_outlet, fulltext).
    examine_lookup = {}

    def store_article_for_lookup(article):
        """Helper to store the article in 'examine_lookup' using build_match_key."""
        title        = article.get("title", "")
        media_outlet = article.get("media_outlet", "")
        fulltext     = article.get("fulltext", "")

        # Build the key, respecting SPECIAL_MATCHES
        key = build_match_key(title, media_outlet, fulltext)

        fields_to_transfer = {}
        if "high_rate_2" in article:
            fields_to_transfer["high_rate_2"] = article["high_rate_2"]
        if "nuance_2" in article:
            fields_to_transfer["nuance_2"] = article["nuance_2"]
        if "comments" in article:
            fields_to_transfer["comments"] = article["comments"]
        if "feedback_2" in article:
            fields_to_transfer["feedback_2"] = article["feedback_2"]

        examine_lookup[key] = fields_to_transfer

    # Populate examine_lookup
    if isinstance(examine_data, list):
        for article in examine_data:
            store_article_for_lookup(article)
    else:
        for article in examine_data.get("articles", []):
            store_article_for_lookup(article)

    # B) Read processed_all_articles_fixed_5.jsonl and update with pre-pipeline fields
    print(f"Reading 5-file: {file_5_jsonl} and updating with pre-pipeline fields...")
    updated_entries_5 = []
    match_count = 0
    unmatched_keys = []

    with open(file_5_jsonl, "r", encoding="utf-8") as f5:
        for line in f5:
            obj_5 = json.loads(line)
            title_5        = obj_5.get("title", "")
            media_outlet_5 = obj_5.get("media_outlet", "")
            fulltext_5     = obj_5.get("clean_fulltext", "")

            match_key = build_match_key(title_5, media_outlet_5, fulltext_5)

            # If there's a match, transfer fields
            if match_key in examine_lookup:
                fields_to_transfer = examine_lookup[match_key]
                for k, v in fields_to_transfer.items():
                    obj_5[k] = v
                match_count += 1
            else:
                unmatched_keys.append((title_5, media_outlet_5, fulltext_5))

            updated_entries_5.append(obj_5)

    print(f"Pre-pipeline match count: {match_count} entries matched.")
    if unmatched_keys:
        print("The following entries in processed_all_articles_fixed_5.jsonl had NO match in the examine file:")
        for key in unmatched_keys:
            title_val, media_outlet_val, cfulltext_val = key
            cfulltext_snippet = cfulltext_val[:50].replace("\n"," ").replace("\r"," ")
            print(f"  Title: {title_val!r}, Media Outlet: {media_outlet_val!r}, Clean Fulltext (snippet): {cfulltext_snippet!r}")

    # Overwrite processed_all_articles_fixed_5.jsonl with updated entries
    with open(file_5_jsonl, "w", encoding="utf-8") as f5_out:
        for entry in updated_entries_5:
            f5_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # --- Main Pipeline Steps -------------------------------------------
    # 1) Copy from processed_all_articles_fixed_5.jsonl to distribution_set.jsonl
    print(f"Copying {file_5_jsonl} to {dist_file}...")
    shutil.copyfile(file_5_jsonl, dist_file)

    # Steps 2-8 in one pass
    print("Beginning main pipeline transformations...")
    with open(file_3_jsonl, "r", encoding="utf-8") as f3, \
         open(dist_file, "r", encoding="utf-8") as dist_in, \
         open(temp_output_file, "w", encoding="utf-8") as dist_out:

        for line3, line_dist in zip(f3, dist_in):
            obj3 = json.loads(line3)      # from file with correct 'date'
            obj_dist = json.loads(line_dist)

            # Step 2: Overwrite date if present
            if "date" in obj3:
                obj_dist["date"] = obj3["date"]

            # Steps 3-5 for clean_fulltext
            if "clean_fulltext" in obj_dist and obj_dist["clean_fulltext"]:
                text = obj_dist["clean_fulltext"]
                # 3) Replace dash-like chars with space
                text = clean_dashes(text)
                # 4) Remove extra spaces
                text = remove_extra_spaces(text)
                # 5) Remove space before punctuation
                text = remove_space_before_punct(text)
                # 6) Rename 'clean_fulltext' -> 'fulltext'
                obj_dist["fulltext"] = text
                del obj_dist["clean_fulltext"]

            # ALSO remove extra spaces and space-before-punctuation in 'title'
            if "title" in obj_dist and obj_dist["title"]:
                t = obj_dist["title"]
                t = remove_extra_spaces(t)
                t = remove_space_before_punct(t)
                obj_dist["title"] = t

            # Steps 7 & 8: Remove unwanted fields and reorder
            obj_dist = remove_unwanted_fields(obj_dist)
            obj_dist = reorder_fields(obj_dist)

            dist_out.write(json.dumps(obj_dist, ensure_ascii=False) + "\n")

    print("Replacing old distribution_set.jsonl with cleaned version...")
    os.replace(temp_output_file, dist_file)

    # --- Create distribution_mortality_subset.jsonl for high_rate_2 == "yes" ---
    print(f"Creating {mortality_subset_file} for high_rate_2 == 'yes'...")
    with open(dist_file, "r", encoding="utf-8") as fin, \
         open(mortality_subset_file, "w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            if data.get("high_rate_2", "").lower() == "yes":
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    print("All steps completed successfully!")
    print(f" - Final data: {dist_file}")
    print(f" - Mortality subset: {mortality_subset_file}")

if __name__ == "__main__":
    main()
