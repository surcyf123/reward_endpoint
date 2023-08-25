import requests
import json
import time

# Endpoint URL
urls= [    
    "http://213.173.102.136:10400", "http://213.173.102.136:10401", "http://213.173.102.136:10402",
    "http://172.17.0.3:30000", "http://172.17.0.3:30001", "http://172.17.0.3:30002", "http://172.17.0.3:30003", "http://172.17.0.3:30004", "http://172.17.0.3:30005", "http://172.17.0.3:30006", "http://172.17.0.3:30007", 
<<<<<<< HEAD
]

url = urls[2]
=======

]

# url = urls[0]
>>>>>>> my_temp_branch

# Data to send
data = {
    "verify_token": "SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n",  # Your authentication token
    "prompt": """
Sure! Here is a summary of the text in seven sentences:\n\nYoichiro Nambu, a Japanese-American physicist, passed away on July 5th at the age of 94. He shared the 2008 Nobel Prize in Physics for his work on spontaneous symmetry breaking. Nambu's research inspired others, including Peter Higgs and François Englert, to develop the theoretical mechanism for the Higgs boson. Higgs and Englert went on to share the 2013 Nobel Prize for their work. Nambu's work was described as ""an act of imagination"" that was ahead of its time by Frank Wilczek. Nambu was born in Tokyo and studied at the Imperial University before moving to the United States. He spent the majority of his career at the University of Chicago and became an American citizen in 1970.\nPrevious Question \nQuestion:Sure! Here's my question:\nGiven that Nambu's work on spontaneous symmetry breaking was so influential and groundbreaking, why do you think it took so long (nearly 50 years) for him to receive recognition in the form of the Nobel Prize?\nAnswer:This question is asking for a step-by-step answer. I will break down the steps as follows:\n\n1. Understand the speaker's message: First, I will read the passage and fully comprehend the speaker's message.\n2. Identify the main points: Once I have understood the message, I will identify the main points that the speaker is trying to convey.\n3. Analyze the information: I will analyze the information and consider how it relates to the question I am being asked.\n4. Provide a step-by-step answer: Based on the analysis, I will provide a step-by-step answer that follows the question asked.\n5. Explain my thoughts: Finally, I will explain my thoughts on the question and why I made the choices that I did.\n\nAsk one relevant and insightful question about the preceding context and previous questions\n
""",
    "responses": [
    '''
Groundbreaking Work: The initial work by Nambu on spontaneous symmetry breaking was indeed groundbreaking. However, groundbreaking work is often ahead of its time, and it can take years, sometimes decades, for the full impact of such work to be understood and appreciated by the wider scientific community.
Experimental Confirmation: Theoretical physics is full of brilliant ideas, but not all of them are correct. It often takes considerable time to gather the experimental evidence needed to confirm or refute theoretical predictions. In Nambu's case, his theories were eventually confirmed by the discovery of the Higgs boson in 2012, which required the construction of the Large Hadron Collider, a project of unprecedented scale in experimental physics.
Nobel Selection Process: The process for awarding the Nobel Prize is notoriously selective and conservative. It often takes many years, sometimes decades, for a discovery to be deemed "Nobel worthy". This is in part because the committee tends to award those whose contributions have stood the test of time and have been thoroughly vetted by the scientific community.
Given the above, it's not unusual that it took nearly 50 years for Nambu's work to be recognized with a Nobel Prize.
As for a related and insightful question: Can you elaborate on the specific contributions Nambu made in the field of spontaneous symmetry breaking, and how these contributions influenced the work of other physicists like Higgs and Englert?
 ''',
 'This is a bad answer that does not relate to the prompt so it will receive lower scores for both reward models and a 0 for the relevance filter, making the total score 0.'
    ]
}

<<<<<<< HEAD
# Make the POST request
=======
>>>>>>> my_temp_branch

for url in urls:
    start_time = time.time()
    response = requests.post(url, json=data)

    print(json.dumps(response.json()))
    time_taken = time.time() - start_time
<<<<<<< HEAD
    print(time_taken)
=======
    print(time_taken)
>>>>>>> my_temp_branch
