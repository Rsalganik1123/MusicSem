HALLUCINATION_PROMPT_BECCA = """# Hallucination Check Prompt for Generated Summary

## Instructions
Evaluate whether the generated summary contains hallucinations based on the provided features/tags from the original source. 
A hallucination is defined as information in the summary that is not present in or contradicts the features from the source material.

## Input Format
- **Original Features/Tags**: [List of key features/tags from the source material]
- **Generated Summary**: [The summary to be evaluated]

## Task
1. Compare each claim or statement in the summary against the original features/tags
2. Identify any information in the summary that:
   - Is not supported by the original features/tags
   - Contradicts the original features/tags
   - Represents an embellishment beyond what can be reasonably inferred
3. **The output should be in JSON format.**

## Output Format
```
{{
"hallucination_detected": [True/False],
}}
```

## Example 1
**Input Data**:
{{
  "original_features": {{
    'situational':  ['3am when Im lonely'],
    'descriptive':['funky', 'jazzy', 'retro vibes', 'distinctive synthesizer in chorus', 'leading bass lines in bridge', 'chill and blended vocals', 'genre of City Pop'],
    'atmospheric'': ['romantic loneliness', 'vulnerability', 'kind of sad in a good way', 'acting heartbroken', 'idealized version of a city'],
    'contextual': ['Plastic Love sounds similar to Once Upon a Night', 'Plastic Love sounds similar to Warm on a Cold Night'],
  }},
  "generated_summary": 'funky, jazzy, retro vibes. I listen to this music at 3am when Im lonely because it romanticizes my loneliness and makes it meaningful. 
    It helps me to enjoy your own loneliness. It has very distinctive synthesizer sounds in the chorus and leading bass lines in the bridge. 
    The vocals are chill and blended.  The genre is like City Pop which describes an idealized version of a city.'  
}}


**Expected Output**:
```
{{
"hallucination_detected": False,
}}

## Example 2
**Input Data**:
{{
  "original_features": {{
    'situational':  ['when I'm quitting my corporate job'],
    'descriptive':['angry punk guitar', 'killer drums', 'harcore vocal processing', 'distortion'],
    'atmospheric'': ['pumped up vibes', 'makes me want to take charge of my life'],
    'contextual': [''],
  }},
  "generated_summary": 'This song makes me happy. It has a soft and exciting vibe with killer drums. I listen to this song at parties or festivals when I feel positive.'
}}

**Expected Output**:
```
{{
"hallucination_detected": True,
}}
```

**Input Data**:
```
{{
  "original_features": {features},
  "generated_summary": {summary}
}}
```
**Expected Output**:
```
"""