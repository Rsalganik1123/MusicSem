DESCRIPTION_PROMPT_BECCA = """
# Summarization task

Write a sentence which combines the associated sentence fragments.
Please do not add anything other than the information given to you.

Your description should:
- Be maximum 4 sentences in length

Your description shouldn't:
- Add any additional information that is not present in the tags
- Include any information that is based on your own knowledge or assumptions

Example: 
  'Situational':  ['3am when Im lonely'],\
  'Descriptive':['funky', 'jazzy', 'retro vibes', 'distinctive synthesizer in chorus', 'leading bass lines in bridge', 'chill and blended vocals', 'genre of City Pop'],\
  'Atmospheric'': ['romantic loneliness', 'vulnerability', 'kind of sad in a good way', 'acting heartbroken', 'idealized version of a city'],\
  'Contextual': ['Plastic Love sounds similar to Once Upon a Night', 'Plastic Love sounds similar to Warm on a Cold Night'],\
  'Metadata': ['funky', 'jazzy', 'retro vibes', 'genre of City Pop']\

  Desired output: This song has funky, jazzy, retro vibes. I listen to this music at 3am when Im lonely because it romanticizes my loneliness and makes it meaningful. \
    It helps me to enjoy your own loneliness. It has very distinctive synthesizer sounds in the chorus and leading bass lines in the bridge. \
    The vocals are chill and blended.  The genre is like City Pop which describes an idealized version of a city.'  \

Tags:

{input_tags}
"""

