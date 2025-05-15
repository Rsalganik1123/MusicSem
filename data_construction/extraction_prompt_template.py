
final_prompt = "Task Description\n \
    You are tasked with analyzing Reddit posts about music and extracting structured information into specific categories. When given a Reddit post discussing music, identify and extract the following: \
    \n Categories to Extract \n \
    Song/Artist pairs (using the names of artists and their songs with unfixed form) some examples - ‘Shake it Off by Taylor Swift’, ‘Radiohead’s Weird Fishes, Genesis - Yes, Maroon5 [She Will Be Loved]\
    Descriptive (using musical attributes) - This includes detailed observations about: \
    - Instrumentation: 'I love the high pass filter on the vocals in the chorus and the soft piano in the bridge' \
    - Production techniques: 'The way they layered those harmonies in the second verse is incredible' \
    - Song structure: 'That unexpected key change before the final chorus gives me goosebumps' \
    - Sound qualities: 'The fuzzy lo-fi beats with that vinyl crackle in the background' \
    - Technical elements: 'The 6/8 time signature makes it feel like its swaying' \
    Contextual (using other songs/artists) - This includes meaningful comparisons such as: \
    - Direct comparisons: 'Sabrina Carpenter\'s *Espresso* is just a mix of old Ariana Grande and 2018 Dua Lipa' \
    - Influences: 'You can tell they\'ve been listening to a lot of Talking Heads' \
    - Genre evolution: 'It\'s like 90s trip-hop got updated with modern trap elements' \
    - Sound-alikes: 'If you like this, you should check out similar artists like...' \
    - Musical lineage: 'They\'re carrying the torch that Prince lit in the 80s' \
    Situational (using an activity, setting, or environment) - This includes relatable scenarios like: \
    - Life events: 'I listened to this song on the way to quitting my sh**ty corporate job' \
    - Regular activities: 'This is my go-to album for late night coding sessions' \
    - Specific locations: 'Hits different when you\'re driving through the mountains at sunset' \
    - Social contexts: 'We always play this at our weekend gatherings and everyone vibes to it' \
    - Seasonal connections: 'This has been my summer anthem for three years running' \
    Atmospheric (using emotions and descriptive adjectives) - This includes evocative descriptions such as: \
    - Emotional impacts: 'This song makes me feel like a manic pixie dream girl in a bougie coffeeshop' \
    - Visual imagery: 'Makes me picture myself in a coming-of-age indie movie, running in slow motion' \
    - Mood descriptions: 'It has this melancholic yet hopeful quality that hits my soul' \
    - Sensory experiences: 'The song feels like a warm embrace on a cold day' \
    - Abstract feelings: 'Gives me this feeling of floating just above my problems' \
    Lyrical (focusing on words and meaning) - This includes thoughtful commentary on: \
    - Storytelling: 'The lyrics tell such a vivid story of lost love that I feel like I\'ve lived it' \
    - Wordplay: 'The clever double entendres in the chorus make me appreciate it more each listen' \
    - Messaging: 'The subtle political commentary woven throughout the verses really resonates' \
    - Personal connection: 'These lyrics seem like they were written about my own life experiences' \
    - Quotable lines: 'That line \'we\'re all just stardust waiting to return\' lives rent-free in my head' \
    Metadata (using information found in labels or research) - This includes interesting facts like: \
    - Technical info: 'The song is hip-hop from the year 2012 with a bpm of 100' \
    - Creation context: 'They recorded this album in a cabin with no electricity using only acoustic instruments' \
    - Chart performance: 'It\'s wild how this underplayed track has over 500 million streams' \
    - Artist background: 'Knowing the guitarist was only 17 when they recorded this makes it more impressive' \
    - Release details: 'This deluxe edition has three bonus tracks that are better than the singles' \
    Sentiment (whether the person feels good or bad about the song) \
    \n Output Format \n \
    Return your analysis as a structured JSON with these categories. {'pairs':[(song_1, artist_1), (song_2, artist_2), ...], 'Descriptive’:[], ‘Contextual’:[], ‘Situational’:[], ‘Atmospheric’:[], ‘Lyrical’: [], ‘Metadata’:[]}. \
    \n Example \n \
    **Input:** \
    'I like Plastic Love by Mariya Takeuchi because of the funky, jazzy, retro vibes. I listen to this music at 3am when Im lonely because it romanticizes my loneliness and makes it meaningful. It helps me to enjoy my own loneliness. It has very distinctive synthesizer sounds in the chorus and leading bass lines in the bridge. The vocals are chill and blended. Another song that sounds very similar is Once Upon a Night by Billyrrom or Warm on a Cold Night by Honne. The genre is like City Pop which describes an idealized version of a city.' \
    **Output:** \
    ```\
    { \
    'pairs': [('Plastic Love', 'Mariya Takeuchi'), ('Once Upon a Night', 'Billyrrom'), ('Warm on a Cold Night', 'HONNE')],\
    'Situational':  ['3am when Im lonely'],\
    'Descriptive':['funky', 'jazzy', 'retro vibes', 'distinctive synthesizer in chorus', 'leading bass lines in bridge', 'chill and blended vocals', 'genre of City Pop'],\
    'Atmospheric'': ['romantic loneliness', 'vulnerability', 'kind of sad in a good way', 'acting heartbroken', 'idealized version of a city'],\
    'Contextual': ['Plastic Love sounds similar to Once Upon a Night', 'Plastic Love sounds similar to Warm on a Cold Night'],\
    'Metadata': ['funky', 'jazzy', 'retro vibes', 'genre of City Pop']\
    }\
    ```\
    ## Data to be analyzed\
    **Input**"

def get_prompt(post:str): 
    return final_prompt + ' ' + post