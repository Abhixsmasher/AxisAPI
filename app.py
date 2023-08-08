from flask import *
import io
import json
import time
import os
import re
from sklearn import preprocessing
import numpy as np
import PyPDF2 as pdf
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from google_drive_downloader import GoogleDriveDownloader as gdd
import nltk
import openai
import time
import smtplib
import ssl
from email.message import EmailMessage

app = Flask(__name__)

nltk.download("stopwords")
tier1=['BITS Pilani',
'DTU',
'NSUT Delhi',
'NIT Tiruchipally',
'NIT Warangal',
'NIT Surathkal',
'Jadavpur University',
'IIIT Allahabad',
'IIT Kharagpur',
'IIT Bombay',
'IIT Madras',
'IIT Kanpur',
'IIT Delhi',
'IIT Guwahati',
'IIT Roorkee',
'IIT Ropar',
'IIT Bhubaneswar',
'IIT Gandhinagar',
'IIT Hyderabad',
'IIT Jodhpur',
'IIT Patna',
'IIT Indore',
'IIT Mandi',
'IIT Varanasi',
'IIT Palakkad',
'IIT Tirupati',
'IIT Dhanbad',
'IIT Bhilai',
'IIT Dharwad',
'IIT Jammu',
'IIT Goa',
'NIT Rourkela',
'IIIT Hyderabad',
'IIIT Delhi']

tier2=['IIIT Bangalore',
'IGDTUW',
'IIITM Gwalior',
'IIIT Lucknow',
'MNNIT Allahabad',
'Punjab Engineering College',
'DAIICT',
'LNMIIT',
'BIT Mesra',
'IIIT Jabalpur',
'Jalpaiguri Government Engineering College',
'IIEST/BESU Shibpur',
'R.V. College of Engineering',
'NIT Bhopal',
'NIT Nagpur',
'NIT Durgapur',
'NIT Jamshedpur',
'NIT Srinagar',
'NIT Allahabad',
'NIT Surat',
'NIT Calicut',
'NIT Jaipur',
'NIT Kurukshetra',
'NIT Silchar',
'NIT Hamirpur',
'NIT Jalandhar',
'NIT Patna',
'NIT Raipur',
'NIT Agartala',
'NIT Arunachal Pradesh',
'NIT Delhi',
'NIT Goa',
'NIT Manipur',
'NIT Meghalaya',
'NIT Mizoram',
'NIT Nagaland',
'NIT Puducherry',
'NIT Sikkim',
'NIT Uttarakhand',
'NIT Andhra Pradesh']

openai.api_key = os.environ["open_ai_key"]

def tokenize(txt):
    tokens= re.split('\W+', txt)
    return tokens

def resume_analysis(file_name):
    f= open(file_name, 'rb')
    reader= pdf.PdfReader(f)
    pg= reader.pages[0]
    txt=pg.extract_text()
    txt=txt.lower()
    resume_vec= tokenize(txt)
    resume_vec = [word for word in resume_vec if not word in stopwords.words()]
    return resume_vec

def avg(a,b):
    return (a+b)/2

def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def url_to_id(url):
    x = url.split("/")
    return x[5]

def generate_interview_questions(job_description):
    prompt = f"Generate 5 tech interview questions, output each question with ### at start, for a role whose job description is:\n {job_description}."
    response = openai.Completion.create(
        engine="text-davinci-002",  # Use appropriate engine (GPT-3) or any upgraded version
        prompt=prompt,
        max_tokens=150,  # Adjust this to control the response length
        stop=None,  # Stop sequences if necessary
        temperature=0.6,  # Adjust this for diversity in responses
        n=1,  # Number of questions to generate
        echo=True,  # Return the prompt in the response for context
    )

    # Extract the generated questions from the API response
    questions = [choice['text'] for choice in response['choices']]

    return questions

def get_question_score(question,response):
    questions=""
    for i in range(len(questions)):
        questions.append(i)
        questions.append(question[i])
    answers=""
    for i in range(len(questions)):
        answers.append(i)
        answers.append(response[i])
    prompt = "The question are: "+questions+" The answers are: "+answers+" Rate the response out of 50 (You can use decimals). \
    The answer should be specific to tech roles. JUST MENTION THE SCORE(just numerical value) \
    AND NOTHING ELSE."
    answer= openai.Completion.create(
        engine="text-davinci-002",  # Use appropriate engine (GPT-3) or any upgraded version
        prompt=prompt,
        max_tokens=150,  # Adjust this to control the response length
        stop=None,  # Stop sequences if necessary
        n=1,  # Number of questions to generate
    )
    score = answer['choices'][0]['text']
    rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", score)
    return rr[0]

@app.route('/CV',methods=['GET'])
def CV_handle():
    JD_text=str(request.args.get('description'))
    JD_text=JD_text.replace(" ", "")
    JD_text=JD_text.lower()
    jd_vec= tokenize(JD_text)
    jd_vec = [word for word in jd_vec if not word in stopwords.words()]
    email=str(request.args.get('email'))
    cgpa=float(request.args.get('cgpa'))
    inst=str(request.args.get('institute'))
    cv=str(request.args.get('CV'))
    ID=url_to_id(cv)
    gdd.download_file_from_google_drive(file_id=ID, dest_path='./lib/data/CV.pdf')
    location= './lib/data/CV.pdf'
    resume_tokens=resume_analysis(location)
    score=jaccard_similarity(jd_vec, resume_tokens)
    score=score*10
    cgpa=cgpa/10
    inst_score=0
    if inst in tier1:
        inst_score=1
    elif inst in tier2:
        inst_score=0.85
    else:
        inst_score=0.75
    final_score=score*inst_score*cgpa
    response={
        'email' : email,
        'CV_score' : final_score
    }
    
    return jsonify(response)

@app.route('/Question',methods=['GET'])
def gen_questions():
    JD_text=str(request.args.get('description'))
    questions = generate_interview_questions(JD_text)
    no_space=questions[0].replace("\n","")
    int_quest=no_space.split("###")
    final_quest=int_quest[2:]
    response={
        'questions' : final_quest 
    }
    return jsonify(response)
    
@app.route('/TestMail',methods=['POST'])
# Set the subject and body of the email
def sendTestMail():
    param=str(request.args.get('email'))
    email_sender = 'hirexs71@gmail.com'
    email_password = 'tcfpjoepyfxyjacd'
    email_receiver = param
    subject = 'Congratulations! You have been shortlisted from HireXS!'
    body = """
    Congratuations! You have been identified 
    """

    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)

# Add SSL (layer of security)
    context = ssl.create_default_context()

# Log in and send the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())
    response={
        "sent" : "true",
    }
    return jsonify(response)
    
@app.route('/Assess',methods=['GET'])
def assess():
    questions=request.args.getlist('questions')
    answers=request.args.getlist('answers')
    email=request.args.get('email')
    score=float(get_question_score(questions,answers))
    response={
        'email': email,
        'score': score
    }
    return jsonify(response)

@app.route('/SelectMail',methods=['POST'])
def sendSelectMail():
    param=str(request.args.get('email'))
    email_sender = 'hirexs71@gmail.com'
    email_password = 'tcfpjoepyfxyjacd'
    email_receiver = param

# Set the subject and body of the email
    subject = 'Congratulations! You have been shortlisted from HireXS!'
    body = """
    Congratuations! You have been selected for interview
    """

    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)

# Add SSL (layer of security)
    context = ssl.create_default_context()

# Log in and send the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())
    response={
        "sent" : "true",
    }
    return jsonify(response)
