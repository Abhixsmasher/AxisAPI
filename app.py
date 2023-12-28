from flask import *
import requests
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
import pymongo
from email.message import EmailMessage
from PIL import Image
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import pandas as pd
import hashlib

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

openai.api_key = os.environ["open_ai_key_1"]

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

def extract_paragraphs_as_json(input_string):
    packages = []
    current_package = None

    # Split input string into lines
    lines = input_string.split('\n')

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check if the line starts with "Package"
        if line.startswith("Package"):
            # If a package is already being processed, append it to the list
            if current_package:
                packages.append(current_package)

            # Initialize a new package
            package_heading = line.split(":")[1].strip()
            current_package = {
                'Heading': package_heading,
                'Paragraphs': [],
            }
        elif current_package:
            # Add the non-empty line to the current package's paragraphs
            current_package['Paragraphs'].append(line)

    # Append the last package to the list
    if current_package:
        packages.append(current_package)

    return json.dumps(packages, indent=2)

def parse_package_string(package_string):
    lines = package_string.split('\n')

    flights = []
    hotel = {}
    estimate_cost = {}

    current_flight = None

    for line in lines:
        if line.startswith('Flight'):
            current_flight = {}
            flights.append(current_flight)
        elif line.startswith('- Departure Airport'):
            current_flight['departure_airport'] = line.split(': ')[1]
        elif line.startswith('- Arrival Airport'):
            current_flight['arrival_airport'] = line.split(': ')[1]
        elif line.startswith('- Airline'):
            current_flight['airline'] = line.split(': ')[1]
        elif line.startswith('- Flight Number'):
            current_flight['flight_number'] = line.split(': ')[1]
        elif line.startswith('Hotel'):
            hotel['name'] = lines[lines.index(line) + 1].split(': ')[1]
            hotel['description'] = lines[lines.index(line) + 2].split(': ')[1]
        elif line.startswith('Estimate Cost'):
            estimate_cost['amount'] = str(line.split(' ')[2].replace(',', ''))
            estimate_cost['currency'] = line.split(' ')[3]

    package_json = {
        'flights': flights,
        'hotel': hotel,
        'estimate_cost': estimate_cost
    }

    return json.dumps(package_json, indent=2)


def get_question_score(question,response):
    questions=""
    for i in range(len(question)):
        questions=questions+str(i)
        questions=questions+question[i]
    answers=""
    for i in range(len(response)):
        answers=answers+str(i)
        answers=answers+response[i]
    prompt = "The question are: "+questions+" The answers are: "+answers+". Score all the responses combined out of 10 combined considering\
    the answers specific to tech roles. JUST MENTION THE FINAL SCORE\
    AND NOTHING ELSE."
    answer= openai.Completion.create(
        engine="text-davinci-002",  # Use appropriate engine (GPT-3) or any upgraded version
        prompt=prompt,
        max_tokens=150,  # Adjust this to control the response length
        stop=None,  # Stop sequences if necessary
        n=1,  # Number of questions to generate
    )
    score = answer['choices'][0]['text']
    rr = re.findall("[+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", score)
    return rr[0]

def get_flights(source, destination, date):
    access_key = '0ec367da1fb2897d377ee3057b944af1'
    start_date= date
    end_date= date
    url = f'http://api.aviationstack.com/v1/flights?access_key={access_key}&dep_iata={source}&arr_iata={destination}&flight_iata=&date_from={start_date}&date_to={end_date}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Request failed with status code: {response.status_code}")
        return "Error!"
    returned_flights=[]
    for i in range(0, 5):
        selected_flight=data['data'][i]
        flight_date=selected_flight['flight_date']
        departure_airport= selected_flight['departure']['airport']
        arrival_airport= selected_flight['arrival']['airport']
        airline= selected_flight['airline']['name']
        flight_number= selected_flight['flight']['iata']
        selected_fight_data= {'flight_date':flight_date,'departure_airport':departure_airport,'arrival_airport':arrival_airport,'airline':airline,'flight_number':flight_number}
        returned_flights.append(selected_fight_data)
    return returned_flights

def get_hotels(destination):
    api_key = "feb69280a932afe21beb5067f434ca4b"
    secret = "4f0bee06e9"
    timestamp = str(int(time.time()))
    signature_data = api_key + secret + timestamp
    x_signature = hashlib.sha256(signature_data.encode()).hexdigest()
    
    headers = {
        'Accept': 'application/json',
        'Api-key': api_key,
        'X-Signature': x_signature
    }

    url = 'https://api.test.hotelbeds.com/hotel-api/1.0/status'

    response = requests.get(url, headers=headers)

    language = 'en'
    headers = {'Api-key': 'feb69280a932afe21beb5067f434ca4b',
              'X-Signature': x_signature ,
              'Accept' : "application/json",
              "Accept-Encoding" : "gzip"}
    url = f'https://api.test.hotelbeds.com/hotel-content-api/1.0/hotels?destinationCode={destination}'


    response = requests.get(url,headers=headers)

    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Request failed with status code: {response.status_code}")
    response=[]
    for i in range(5):
        temp={}
        temp['name']=data['hotels'][i]['name']['content']
        temp['description']=data['hotels'][i]['description']['content']
        response.append(temp)
    return response
@app.route('/makepack',methods=['GET','POST'])
def get_packages():
    source=str(request.args.get('source'))
    destination=str(request.args.get('destination'))
    date1=str(request.args.get('date1'))
    date2=str(request.args.get('date2'))
    event=int(request.args.get('event'))
    going_flights= get_flights(source, destination, date1)
    coming_flights= get_flights(destination, source, date2)
    hotels= get_hotels(destination)
    prompt_medical=f"""
    The details for going flights are:
    {going_flights}
    
    The details for coming back flights are:
    {coming_flights}
    
    The hotel details are:
    {hotels}
    
    The details are given to you in the form of a python list of dictionaries. 
    You are given 5 flights for going to the destination, 5 flights to come back and 5 hotels each. We have to derive a MEDICAL TREATMENT TRIP package using one of these flights and and one of these hotels for the difference between departure and arrival dates. 
    Now what I want you to do is create 2 such packages and describe them in human text. 
    Use around 100 words to describe EACH package. 
    Describe everything about the package from which airline's flight the customer will be taking for travel to and from the destination.
    ALSO KEEP IN MIND THAT I ONLY WANT THE DESCRIPTIONS AND NO OTHER TEXT IN YOUR RESPONSE. 
    MAKE SURE YOU ELABORATE ON THE EXCLUSIVE MEDICAL FACILITIES THE DESTINATION HAS TO OFFER.
    DO NOT FORGET TO USE THE FLIGHT INFORMATION GIVEN TO YOU FOR BOTH GOING TO THE DESTINATION AND COMING BACK. PUT THAT IN THE PACKAGE DESCRIPTION ALSO.


    INSTEAD OF USING 'PACKAGE 1' AND 'PACKAGE 2', USE A CATCHY TITLE FOR IT AND THEN ADD THE PACKAGE NUMBER AT THE END BUT USE THIS ONLY IN THE TITLE.
    """
    prompt_business=f"""
    The details for going flights are:
    {going_flights}
    
    The details for coming back flights are:
    {coming_flights}
    
    The hotel details are:
    {hotels}
    
    The details are given to you in the form of a python list of dictionaries. 
    You are given 5 flights for going to the destination, 5 flights to come back and 5 hotels each. We have to derive a BUSINESS TRIP package using one of these flights and and one of these hotels for difference between departure and arrival dates. 
    Now what I want you to do is create 2 such packages and describe them in human text. 
    Use around 100 words to describe EACH package. 
    Describe everything about the package from which airline's flight the customer will be taking for travel to and from the destination.
    ALSO KEEP IN MIND THAT I ONLY WANT THE DESCRIPTIONS AND NO OTHER TEXT IN YOUR RESPONSE. 
    MAKE SURE YOU ELABORATE ON HOW THE DESTINATION WOULD BE HELPFUL IN BUSINESS MEETINGS AND THE BUSINESS OPPORTUNITIES IT HAS TO OFFER.
    DO NOT FORGET TO USE THE FLIGHT INFORMATION GIVEN TO YOU FOR BOTH GOING TO THE DESTINATION AND COMING BACK. PUT THAT IN THE PACKAGE DESCRIPTION ALSO.


    INSTEAD OF USING 'PACKAGE 1' AND 'PACKAGE 2', USE A CATCHY TITLE FOR IT AND THEN ADD THE PACKAGE NUMBER AT THE END BUT USE THIS ONLY IN THE TITLE.
    """
    prompt_vacation=f"""
    The details for going flights are:
    {going_flights}
    
    The details for coming back flights are:
    {coming_flights}
    
    The hotel details are:
    {hotels}
    
    The details are given to you in the form of a python list of dictionaries. 
    You are given 5 flights for going to the destination, 5 flights to come back and 5 hotels each. We have to derive a destination holiday vacation package using one of these flights and and one of these hotels for difference between departure and arrival dates. 
    Now what I want you to do is create 2 such packages and describe them in human text. 
    Use around 100 words to describe EACH package. 
    Describe everything about the package from which airline's flight the customer will be taking for travel to and from the destination.
    ALSO KEEP IN MIND THAT I ONLY WANT THE DESCRIPTIONS AND NO OTHER TEXT IN YOUR RESPONSE. 
    MAKE SURE YOU ELABORATE ON THE EXCLUSIVE TOURISM SPOTS AND FACILITIES THE CITY HAS TO OFFER.
    DO NOT FORGET TO USE THE FLIGHT INFORMATION GIVEN TO YOU FOR BOTH GOING TO THE DESTINATION AND COMING BACK. PUT THAT IN THE PACKAGE DESCRIPTION ALSO.
    USE EMOJIS EXTENSIVELY IN THE HEADING AND DESCRIPTION ALSO


    INSTEAD OF USING 'PACKAGE 1' AND 'PACKAGE 2', USE A CATCHY TITLE FOR IT AND THEN ADD THE PACKAGE NUMBER AT THE END BUT USE THIS ONLY IN THE TITLE.
    """
    prompt_weddings=f"""
    The details for going flights are:
    {going_flights}
    
    The details for coming back flights are:
    {coming_flights}
    
    The hotel details are:
    {hotels}
    
    The details are given to you in the form of a python list of dictionaries. 
    You are given 5 flights for going to the destination, 5 flights to come back and 5 hotels each. We have to derive a destination wedding package using one of these flights and and one of these hotels for difference between departure and arrival dates. 
    Now what I want you to do is create 2 such packages and describe them in human text. 
    Use around 100 words to describe EACH package. 
    Describe everything about the package from which airline's flight the customer will be taking for travel to and from the destination.
    ALSO KEEP IN MIND THAT I ONLY WANT THE DESCRIPTIONS AND NO OTHER TEXT IN YOUR RESPONSE. 
    MAKE SURE YOU ELABORATE ON THE EXCLUSIVE WEDDING FACILITIES THE CITY HAS TO OFFER.
    DO NOT FORGET TO USE THE FLIGHT INFORMATION GIVEN TO YOU FOR BOTH GOING TO THE DESTINATION AND COMING BACK. PUT THAT IN THE PACKAGE DESCRIPTION ALSO.
    USE EMOJIS EXTENSIVELY IN THE HEADING AND DESCRIPTION ALSO


    INSTEAD OF USING 'PACKAGE 1' AND 'PACKAGE 2', USE A CATCHY TITLE FOR IT AND THEN ADD THE PACKAGE NUMBER AT THE END BUT USE THIS ONLY IN THE TITLE.
    """

    if event==0:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"{prompt_weddings}"}
            ]
        )
    elif event==1:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"{prompt_vacation}"}
            ]
        )
    elif event==2:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"{prompt_business}"}
            ]
        )
    else:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"{prompt_medical}"}
            ]
        )
    speech=response.choices[0].message['content']
    print(speech)
    tt=extract_paragraphs_as_json(speech)
    print(tt)
    return jsonify(json.loads(tt))
    
@app.route('/packdetail',methods=['GET','POST'])    
def get_package_details():
    package=str(request.args.get('package'))
    prompt_package_details= f"""
        The description of the package is:
        {package}

        I want you to use this description and generate package details in the format given below:
        
        
            <Package Title>
            
            Flight 1:
            - Departure Ariport: ........
            - Arrival Airport: ........ 
            - Airline: ........
            - FLight Number: ........

            Flight 2:
            - Departure Ariport: ........
            - Arrival Airport: ........
            - Airline: ........
            - Flight Number: ........


            Hotel:
            - Name:........
            - Description:........
            
            Exclusive Details:
            
            <put random facts about the destination related to the package. You can pick it up from the description given above>
            
            Estimate Cost: <put rounded off number that estimates the total cost of package between 2000 and 2000000> INR


           
        
        Replace the dots with the information you can retreive from the description given. 
        DO NOT USE RUPEES SYMBOL OR ANY SYMBOL WITH NUMBERS WHEN SHOWING ESTIMATE COST.
        ALSO KEEP IN MIND THAT I ONLY WANT THE PACKAGE DETAILS AND NO OTHER TEXT IN YOUR RESPONSE. 
        ONLY GIVE ME TEXT REPSPONSE.
        IT IS ALSO MANDATORY FOR YOU TO REPLACE ALL DOTS. DO NOT LEAVE ANY IN THE RESPONSE YOU GIVE. IF ANY SUCH DETAIL IS NOT GIVEN IN THE DESCRIPTION, MAKE IT UP YOURSELF.
        DO NOT PUT ANYTHING IN CURLY BRACKETS IN THE OUTPUT YOU GIVE. I WANT COMPLETE ANSWER!
        EXTENSIVELY USE EMOJIS IN THE OUTPUT TO MAKE IT LOOK BETTER!
        DO NOT FORGET TO USE EMOJIS!
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{prompt_package_details}"}
        ]
    )
    pack=response.choices[0].message['content']
    json_output = parse_package_string(pack)
    return jsonify(json.loads(json_output))
    
@app.route('/plot',methods=['GET','POST'])
def plotcsv():
    csv=str(request.args.get('csv'))
    ID=url_to_id(csv)
    gdd.download_file_from_google_drive(file_id=ID, dest_path='./lib/data/data.csv')
    time.sleep(3)
    df = pd.read_csv('./lib/data/data.csv')
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            try:
                df[column] = pd.to_numeric(df[column])
            except ValueError:
                df = df.drop(columns=[column])
    correlations = df.corr()
    top_correlations = correlations.unstack().sort_values(ascending=False).drop_duplicates()[:3]
    plot_data = {}
    for i, (param1, param2) in enumerate(top_correlations.index):
        grouped = df.groupby(param1).mean().reset_index()
        data = {
            'x': grouped[param1].tolist(),
            'y': grouped[param2].tolist(),
            'xlabel': param1,
            'ylabel': param2,
            'title': f"{param2} vs {param1}",
            'type': 'bar'
        }
        plot_data[f'correlation_{i + 1}'] = data
    plot_data['ID']=111
    client = pymongo.MongoClient("mongodb+srv://mahirakajaria:NL1htAGffe0TLscA@cluster0.estoffi.mongodb.net/")  # Replace with your MongoDB connection URL

# Specify the database and collection names
    db_name = "test"
    collection_name = "graphs"

# Access the database
    db = client[db_name]

# Drop the existing collection if it exists
    if collection_name in db.list_collection_names():
        db[collection_name].drop()

# Create a new collection and insert the data
    collection = db[collection_name]
    collection.insert_one(plot_data)

# Close the MongoDB connection
    client.close()
    os.remove('./lib/data/data.csv')
    final={
        'status':'success'
    }
    return jsonify(final)

@app.route('/csvanalyze',methods=['GET','POST'])
def csvanalyze():
    csv=str(request.args.get('csv'))
    query=str(request.args.get('query'))
    ID=url_to_id(csv)
    gdd.download_file_from_google_drive(file_id=ID, dest_path='./lib/data/data.csv')
    time.sleep(3)
    agent = create_csv_agent(
    ChatOpenAI(temperature=0,openai_api_key=os.environ["open_ai_key_1"], model="gpt-3.5-turbo-0613"),
    './lib/data/data.csv',
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    answer=agent.run(f'Analyse the data and generate top 3 strategies to {query} AND MENTION SPECIFIC PRODUCTS \
    that are to be affected in 10 words each. Also display the chance of success for each out of 100%. The strategies should \
    be like the following example:\
    1. Increase visibility of low-selling products (Item_Visibility) by placing them in prominent areas. Chance of success: 80%\
       - Products: FDX07, NCD19, DRI11')
    time.sleep(3)
    prompt=f"For the given text :{answer} , separate each strategy into a json type format with each strategy as an element of list and with products affected and chance of succes as sepatate elements of that element."
    gptquery = openai.Completion.create(
        engine="text-davinci-002",  # Use appropriate engine (GPT-3) or any upgraded version
        prompt=prompt,
        max_tokens=500,
    )
    strategies=gptquery['choices'][0]['text']
    actual=json.loads(strategies)
    mong_dict={
        "strategies" : actual,
        "ID" : 222,
    }
    client = pymongo.MongoClient("mongodb+srv://mahirakajaria:NL1htAGffe0TLscA@cluster0.estoffi.mongodb.net/")  # Replace with your MongoDB connection URL
    db_name = "test"
    collection_name = "strats"
    
# Access the database
    db = client[db_name]

# Drop the existing collection if it exists
    if collection_name in db.list_collection_names():
        db[collection_name].drop()

# Create a new collection and insert the data
    collection = db[collection_name]
    collection.insert_one(mong_dict)

# Close the MongoDB connection
    client.close()
    response={
        'status': 'success',
    }
    os.remove('./lib/data/data.csv')
    return jsonify(response)
    
@app.route('/emailpost',methods=['GET','POST'])
def email_post():
    api_key = "bb_pr_ed5ba364d7e725f3744ea0f1fb2556"
    headers = {
      'Authorization' : f"Bearer {api_key}"
    }
    data={
      "template": "20KwqnDEry0Qbl17dY",
      "modifications": [
        {
          "name": "additional_text",
          "text": request.args.get('additionaltext'),
          "color": None,
          "background": None
        },
        {
          "name": "offer_title",
          "text": request.args.get('offertitle'),
          "color": None,
          "background": None
        },
        {
          "name": "subtitle",
          "text": request.args.get('subtitle'),
          "color": None,
          "background": None
        },
        {
          "name": "offer_image",
          "image_url": request.args.get('productimg'),
        },
        {
          "name": "validity_date",
          "text": request.args.get('valdate'),
          "color": None,
          "background": None
        },
        {
          "name": "Company logo",
          "image_url": request.args.get('companyimg')
        }
      ],
      "webhook_url": None,
      "transparent": False,
      "metadata": None
    }
    response=requests.post('https://api.bannerbear.com/v2/images',
                      json=data,headers=headers)
    time.sleep(3)
    gen_id=response.json()['uid']
    link='https://api.bannerbear.com/v2/images/'+gen_id
    response=requests.get(link,headers=headers)
    response_json=response.json()
    img_down=response_json['image_url']
    data = requests.get(img_down).content
    time.sleep(3)
    f = open('img.jpg','wb')
    f.write(data)
    f.close()
    receivers_mail = request.args.getlist('receivers')
    email_to = ", ".join(receivers_mail)
    file_path = 'img.jpg'
    email_subject = request.args.get('offertitle')
    email_host = 'smtp.gmail.com'
    email_port = 587
    email_username = 'hirexs635@gmail.com'
    email_password = 'jxjyquwhgmswhvax'
    server = smtplib.SMTP(email_host, email_port)
    server.ehlo()
    server.starttls()
    server.login(email_username, email_password)
    msg = MIMEMultipart()
    msg['From'] = email_username
    msg['To'] = email_to
    msg['Subject'] = email_subject
    with open(file_path, 'rb') as f:
        file_data = f.read()
        file_name = f.name.split('/')[-1]
        attachment = MIMEApplication(file_data, name=file_name)
        msg.attach(attachment)
    body = 'Find attractive offers for you below'
    msg.attach(MIMEText(body, 'plain'))
    server.sendmail(email_username, email_to, msg.as_string())
    server.quit()
    os.remove('img.jpg')
    response={
        'status': 'success',
    }
    return jsonify(response)
    
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

@app.route('/Paraphrasejd',methods=['GET','POST'])
def paraphrase():
    JD_text=str(request.args.get('description'))
    role=str(request.args.get('role'))
    prompt="For the role of "+role+" and the following JD, rewrite it to suit better to the role and \
    PRINT ONLY THE MODIFIED JD WITH ONLY THE SPECIFICATIONS AND ROLE MENTIONED WHILE REPLACING THE % WITH SPACE as the JD is coming\
    from a URL: "+ JD_text
    response = openai.Completion.create(
        engine="text-davinci-002",  # Use appropriate engine (GPT-3) or any upgraded version
        prompt=prompt,
        max_tokens=len(JD_text),
        stop = ["input:"],
    )

    # Extract the generated questions from the API response
    New_JD = response['choices'][0]['text']
    ans={
        "JD": New_JD
    }
    return jsonify(ans)
    
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
    
@app.route('/TestMail',methods=['GET','POST'])
# Set the subject and body of the email
def sendTestMail():
    param=str(request.args.get('email'))
    job_id=str(request.args.get('job_id'))
    print(job_id)
    email_sender = 'hirexs71@gmail.com'
    email_password = 'tcfpjoepyfxyjacd'
    email_receiver = param
    link=''
    if job_id == "6789":
        link = 'https://senior-software-engineer-bot-app-ixyu3x2ys72js3uwpnypgp.streamlit.app/'
    elif job_id == "9023":
        link = 'https://appuct-manager-bot-app-5ktvwc47vhuhwkhow4xdxt.streamlit.app/'
    elif job_id == "5214":
        link = 'https://data-scientist-bot-app-txnu96cu868jnqsxmtcx57.streamlit.app/'
    elif job_id == "7532":
        link = 'https://financial-advisor-bot-app-nttgg5tjvjxup9wxiv6vzh.streamlit.app/'
    elif job_id == "8346":
        link = 'https://software-engineer-bot-app-ec7xda7ny2wx7bilrgbztf.streamlit.app/'
    elif job_id == "1467":
        link = 'https://ai-research-scientist-bot-app-n9ic9sbwabn92y9z5vjpdd.streamlit.app/'
    else:
        link = 'https://software-engineer-bot-app-ec7xda7ny2wx7bilrgbztf.streamlit.app/'
    subject = ' Assessment Link from HireXS - Urgent Completion Required'
    body = f"""
    Dear Candidate,
    We hope this email finds you well. We are excited to inform you that we have identified you as a promising candidate from HireXS.
    As part of our rigorous selection process, we kindly request your assistance in evaluating your suitability for this position.
    Please complete the below assesment within a day so that we can further the evaluation process.
    Assesment Link:- {link}  
    Remember to open the link in a new tab."""

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

@app.route('/SelectMail',methods=['GET','POST'])
def sendSelectMail():
    param=str(request.args.get('email'))
    email_sender = 'hirexs71@gmail.com'
    email_password = 'tcfpjoepyfxyjacd'
    email_receiver = param

# Set the subject and body of the email
    subject = 'Congratulations on Your Selection for an Interview at Axis Bank!'
    body = """
    Dear Candidate,
    We are thrilled to inform you that after a thorough review of your application, we are impressed with your qualifications and experiences,
    and we would like to invite you for an interview at Axis Bank.
    Your application stood out among a competitive pool of candidates, and we believe your skills and background align well with what we are looking
    for in this position.
    
    Interview Details:
    Date: 15th September 2023
    Time: 10:00 AM
    Location: Axis Bank, Bangalore
    
    Best regards,

    Axis Bank
    From HireXS Portal.
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

@app.route('/submit',methods=['GET','POST'])
def save_to_mongodb():
    email= str(request.args.get('email'))
    email=email.replace('%40','@')
    score= request.args.get('score')
    job_id= str(request.args.get('job_id'))
    client = pymongo.MongoClient("mongodb+srv://mahirakajaria:NL1htAGffe0TLscA@cluster0.estoffi.mongodb.net/")
    db = client["test"]
    user_collection = db['users']
    cvs_collection = db['cvs']
    user = user_collection.find_one({'email': email})
    
    if user:
        user_id = user['_id']
        cvs_collection.update_one(
            {'jobId': job_id, 'owner': user_id},
            {"$set": {'testScore': score}}
        )
        client.close()
    else:
        print("Candidate email not found.")
        client.close()
    response={
        "val":"None",
        "email":email,
        'testscore': score,
        'job_id':job_id,
    }
    return jsonify(response) 
