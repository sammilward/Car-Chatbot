#######################################################
# Initialise Wikipedia agent
#######################################################
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('en')
wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)

#######################################################
#  Initialise AIML agent
#######################################################
RESPONSEAGENT = 'aiml'
SIMILARITYTHRESHOLD = 0.5
NUMBEROFGOOGLEPLACESTODISPLAY = 5

AIMLfile = "CarChatbotAIML.xml"

import aiml
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles=AIMLfile)

#######################################################
# Parse XML to get all AIML questions not including '*'
#######################################################
from xml.dom import minidom
xmldoc = minidom.parse(AIMLfile)
itemlist = xmldoc.getElementsByTagName('pattern')

sentences = []

for sentence in itemlist:
    if '*' not in sentence.childNodes[0].nodeValue:
        sentences.append(str(sentence.childNodes[0].nodeValue).strip())

#######################################################
# Initialise Sklearn TfidfVectorizer
#######################################################
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = ("a", "the", "an", "is", "are", "me", "tell")
vectorizer = TfidfVectorizer(stop_words = stop_words)

#######################################################
# GoogleMaps
#######################################################
import googlemaps
import json

GoogleAPIKey = "EnterAPIKey"
gmaps = googlemaps.Client(key=GoogleAPIKey)

#######################################################
# GeoCoder
#######################################################
import geocoder

g = geocoder.ip('me')
latlon = {"lat" : g.latlng[0],
          "lng" : g.latlng[1]}

#######################################################
# Image Classification
#######################################################
import os
import cv2
import wget
from tensorflow.keras import models, backend, layers
import tkinter
from tkinter.filedialog import askopenfilename

#######################################################
# Initialise NLTK agent
#######################################################
import nltk

v = """
broken_engine => {}
flat_battery => {}
smashed_windshield => {}
poor_suspension => {}
flickering_headlight => {}
bad_handling => {}
slow_acceleration => {}
broken_fog_light => {}

car1 => c1
car2 => c2
car3 => c3
car4 => c4
car5 => c5
car6 => c6
car7 => c7
car8 => c8
car9 => c9
car10 => c10

be_in => {}
"""

#First Order Logic valuation
NUMBEROFCARS = 10
folval = nltk.Valuation.fromstring(v)
grammar_file = 'simple-sem.fcfg'
objectCounter = 0
carNameDict = {}

#######################################################
# Transformer
#######################################################

from squad_transformer import SquadTransformer
transformer = SquadTransformer()

#######################################################
# Reinforcement Learning
#######################################################

import gym
import os
import numpy as np
from IPython.display import clear_output
from time import sleep
import random

QTABLEFILENAME = "QTable.npy"

env = gym.make("Taxi-v3").env

alpha = 0.1
gamma = 0.6
epsilon = 0.1

#######################################################
# COMMANDS
#######################################################

ENDSESSIONCMD = 0
WIKICMD = 1
GOOGLEGETPLACESCMD = 2
GOOGLEGETPLACEDETAILSCMD = 3
GOOGLETRAVELDISTANCECMD = 4
GOOGLETRAVELTIMECND = 5
MODELCLASSIFICATIONCMD = 6
CHOOSEMODELIMAGECMD = 7
MANUFACTURERCLASSIFICATIONCMD = 8
CHOOSEMANUFACTURERIMAGECMD = 9
MODELANDMANUFACTURERCALASSIFICATIONCMD = 10
CHOOSEMODELANDMANUFACTURERIMAGECMD = 11
SETCARHASISSUE = 12
FIXCARISSUE = 13
DOESCARHAVEISSUE = 14
SETCARNAME = 15
WHATISWRONGWITH = 16
GETCARNAME = 17
RESETCAR = 18
RESETALL = 19
WHICHCARSHAVE = 20
LISTALLISSUES = 21
TRAINTAXIGAME = 22
TRAINTAXIGAMEXTIMES = 23
PLAYTAXIGAME = 24
CLEARTAXIGAMETRAINING = 25
EXPLAINTAXIGAME = 26
UNKNOWNCMD = 99

#######################################################
# Image Classification Class
#######################################################

class Image_Predictor:
    def __init__(self, car_model_model_location, car_manufacturer_model_location):
        self.load_Models(car_model_model_location, car_manufacturer_model_location)
        self.set_class_names()
        
    def load_Models(self, car_model_model_location, car_manufacturer_model_location):
        self.model_for_car_model = models.load_model(car_model_model_location)
        self.model_for_car_manufacturer = models.load_model(car_manufacturer_model_location)

    def set_class_names(self):
        self.model_class_names = ['300 SRT-8 2010', 'Continental GT Coupe 2007', 'PT Cruiser Convertible 2008', 'Savana Van 2012', 'Traverse SUV 2012']
        self.manufacturer_class_names = ['Bentley', 'Chevrolet', 'Chrysler', 'GMC', 'Lamborghini']

    def predict_car_model(self, img):
        predictions = self.model_for_car_model.predict(img)
        return self.model_class_names[np.argmax(predictions)]

    def predict_car_manufacturer(self, img):
        predictions = self.model_for_car_manufacturer.predict(img)
        return self.manufacturer_class_names[np.argmax(predictions)]

#######################################################
# Chatbot Functions
#######################################################

def GetUserInput():
    try:
        userInput = input("Human: ")
        return userInput
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        exit()

def UserSaysYes(userInput):
    if userInput.casefold() == "y".casefold() or userInput.casefold() == "ye".casefold() or userInput.casefold() == "yes".casefold():
        return True
    else:
        return False

def ChatBotPrint(output):
    print("Chatbot: " + output)

def IsCommand(AIMLResponse):
    return AIMLResponse[0] == '#'

def GetCommand(AIMLResponse):
    return int(AIMLResponse[1:].split('$')[0])

def GetValues(AIMLResponse):
    return AIMLResponse[1:].split('$')

def DisplayWikiResponse(answer, userInput):
    values = GetValues(answer)
    searchTerm = values[1]
    WikiPage = InvokeWiki(searchTerm)
    if WikiPageExists(WikiPage):
        AskUserToUseWiki(WikiPage)
    else:
        transformer_predict(userInput)

def InvokeWiki(term):
    return wiki_wiki.page(term)

def WikiPageExists(wpage):
    if wpage.exists():
        return True
    else:
        return False

def AskUserToUseWiki(wpage):
    ChatBotPrint("Im not sure, would you like me to search wikipedia for you?")
    userInput = GetUserInput()
    if UserSaysYes(userInput):
        ChatBotPrint(wpage.summary)
        ChatBotPrint("Learn more at " + wpage.canonicalurl)
    else:
        ChatBotPrint("How else can I help?")

def handle_unknown_input(userInput):
    similarAnswer = GetSimilarAnswer(userInput)
    if(similarAnswer):
        PostProcessResponse(similarAnswer, userInput)
    else:
        transformer_predict(userInput)

def GetSimilarAnswer(userInput):
    UserSentanceSimilarityArray = GetUserInputSimilarityArray(userInput)
    if GetSimilarityScore(UserSentanceSimilarityArray) > SIMILARITYTHRESHOLD:
        return kern.respond(GetSimilarSentence(UserSentanceSimilarityArray)) #Get the Response to the closest sentence

def GetUserInputSimilarityArray(userInput):
    sentencesWithUserInput = sentences + [userInput] #Add users input to sentences list, so it can find a similar sentence from the AIML patterns/responces
    matrix = vectorizer.fit_transform(sentencesWithUserInput) #Get Tf-idf document-term matrix
    similarityArray = cosine_similarity(matrix) #Get array of cosineSimilarity
    UserSentanceSimilarityArray = similarityArray[len(similarityArray)-1][0:len(similarityArray)-1] #Get the array of similarity percentages for the users input
    #Removing the last element of the array, as this is the similarity of the users input vs the users input, which returns 1.  
    return UserSentanceSimilarityArray

def GetSimilarityScore(UserSentanceSimilarityArray):
    percentageSimilarity = UserSentanceSimilarityArray.max() #Get the max value in the array to find the similarity score.
    return percentageSimilarity

def GetSimilarSentence(UserSentanceSimilarityArray):
    mostSimilarSentenceIndex = np.argmax(UserSentanceSimilarityArray) #find the index of the highest similarity score.
    return sentences[mostSimilarSentenceIndex]

def GetGooglePlacesResponse(searchQuery):
    return gmaps.places(query=searchQuery)

def DisplayGooglePlaces(jsonResponse):
    ChatBotPrint("Here are some places near you.")
    for i, place in enumerate (jsonResponse["results"], start=1):
        if i == (NUMBEROFGOOGLEPLACESTODISPLAY+1):
            break
        print(str(i) + ".")
        print(place["name"])
        if "formatted_address" in place:
            ChatBotPrint("Address: " + place["formatted_address"])
        if "opening_hours" in place:
            ChatBotPrint("Open Now? " + str(place["opening_hours"]["open_now"]))

def GetGoogleSinglePlaceResponse(placeId):
    return gmaps.place(placeId)

def ParseGooglePlaceId(jsonResponse):
    return jsonResponse["results"][0]["place_id"]

def DisplayExtraPlaceDetails(jsonResponse):
    if "formatted_phone_number" in jsonResponse["result"]:
        ChatBotPrint("Phone number: " + str(jsonResponse["result"]["formatted_phone_number"]))

    if "rating" in jsonResponse["result"]:
        ChatBotPrint("Rating: " + str(jsonResponse["result"]["rating"]))

    if "website" in jsonResponse["result"]:
        ChatBotPrint("Website: " + str(jsonResponse["result"]["website"]))

def GetGoogleDistanceAndTravelTime(origin, place_id):
    return gmaps.distance_matrix(origin, "place_id:" + place_id)

def DisplayTravelTimeToPlace(jsonResponse):
    if "duration" in jsonResponse["rows"][0]["elements"][0]:
        ChatBotPrint("Duration: " + str(jsonResponse["rows"][0]["elements"][0]["duration"]["text"]))

def DisplayTravelDistanceToPlace(jsonResponse):
    if "distance" in jsonResponse["rows"][0]["elements"][0]:
        ChatBotPrint("Distance: " + str(jsonResponse["rows"][0]["elements"][0]["distance"]["text"]))

def get_image_array(source):
    IMG_HEIGHT = 300
    IMG_WIDTH = 300
    COLOUR_CHANNELS = 3

    if("  #99$" in source):
        source = source.replace("  #99$", ".")
    if("$" in source):
        source = source.split('$')[1]

    if os.path.exists(source):
        raw_image = cv2.imread(source)
    else:
        try:
            fileName = source.split("/")[-1]
            raw_image = wget.download(source)
            raw_image = cv2.imread(fileName)
            print("") #Adds a new line after the download, easier for users to read
            os.remove(fileName)
        except Exception:
            return None

    input_shape = (IMG_HEIGHT, IMG_WIDTH, COLOUR_CHANNELS)

    scaled_image = cv2.resize(raw_image, dsize=(IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite("scaled.jpg", scaled_image)

    scaled_image = scaled_image/255
    image_array = scaled_image.reshape(1, IMG_HEIGHT, IMG_WIDTH, COLOUR_CHANNELS)

    os.remove("scaled.jpg")

    return image_array

def ShowImageSelector():
    root = tkinter.Tk()
    root.withdraw()
    filename = askopenfilename()
    return filename

def does_image_exist(img):
    if (img is None):
        ChatBotPrint("I could not retrieve this image")
        return False
    else:
        return True

def issue_to_human_readable(issue):
    return issue.lower().replace("_", " ")

def human_readable_to_issue(humanReadable):
    return humanReadable.lower().replace(" ", "_")

def does_issue_already_exist(car, issue):
    issueAlreadyExists = False
    issueList = folval[issue]
    for element in issueList:
        if (element[0], get_car_abbreviation(car)) in folval['be_in']:
            issueAlreadyExists = True
            break
    return issueAlreadyExists

def set_car_issue(car, humanReadableIssue):
    try:
        issue = human_readable_to_issue(humanReadableIssue)
        issueAlreadyExists = does_issue_already_exist(car, issue)

        if issueAlreadyExists == False:
            global objectCounter
            issueId = 'i' + str(objectCounter) #objectNumber
            objectCounter += 1
            folval['i' + issueId] = issueId
            if len(folval[issue]) == 1:
                if ('',) in folval[issue]:
                    folval[issue].clear()
            folval[issue].add((issueId,))  # insert type of issue
            if len(folval["be_in"]) == 1:
                if ('',) in folval["be_in"]:
                    folval["be_in"].clear()
            folval["be_in"].add((issueId, folval[car]))
            ChatBotPrint("Issue logged successfully")
        else:
            ChatBotPrint("This issue is already logged for this car")
    except nltk.sem.evaluate.Undefined:
        print(f'Sorry I can\'t handle {car} or {humanReadableIssue}')

def get_car_abbreviation(car):
    return car[0] + car[3:]

def get_car_name_from_abbreviation(abbreviation):
    return 'car' + abbreviation[1:]

def remove_car_issue(car, humanReadableIssue):
    try:
        removed = False
        issue = human_readable_to_issue(humanReadableIssue)
        carabrev = get_car_abbreviation(car)

        issueExists = does_issue_already_exist(car, issue)
        if issueExists == True:
            issueList = folval[issue]

            for issueElement in issueList:
                issueNumber = issueElement[0]
                be_in_element = (issueNumber, carabrev)

                if be_in_element in folval['be_in']:
                    issueConstant = 'i' + issueNumber
                    folval['be_in'].remove(be_in_element)
                    folval[issue].remove((issueNumber,))
                    del folval[issueConstant]
                    break
            removed = True
        else:
            removed = False
        return removed
    except nltk.sem.evaluate.Undefined:
        print(f'Sorry I can\'t handle {car} or {humanReadableIssue}')
    
def does_car_have_issue(car, humanReadableIssue):
    try:
        issue = human_readable_to_issue(humanReadableIssue)
        g = nltk.Assignment(folval.domain)
        m = nltk.Model(folval.domain, folval)
        sent = 'a ' + issue + ' is_in ' + car
        results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
        if results[2] == True:
            ChatBotPrint("Yes.")
        else:
            ChatBotPrint("No.")
    except ValueError:
        print(f'Sorry I can\'t handle {car} or {humanReadableIssue}')

def what_is_wrong_with(car):
    try:
        g = nltk.Assignment(folval.domain)
        m = nltk.Model(folval.domain, folval)
        e = nltk.Expression.fromstring("be_in(x," + car + ")")
        sat = m.satisfiers(e, "x", g)
        if len(sat) == 0:
            ChatBotPrint("No issues logged.")
        else:
            sol = folval.values()
            for so in sat:
                for k, v in folval.items():
                    if len(v) > 0:
                        vl = list(v)
                        if len(vl[0]) == 1:
                            for i in vl:
                                if i[0] == so:
                                    ChatBotPrint(issue_to_human_readable(k))
                                    break
    except nltk.sem.evaluate.Undefined:
        print(f'Sorry I can\'t handle {car}')

def reset_car(car):
    issuesToRemove = []
    carabrev = get_car_abbreviation(car)
    
    if car in carNameDict:
        del carNameDict[car]

    be_in_list = folval['be_in']
    for element in be_in_list:
        if element == ('',):
            break
        if element[1] == carabrev:
            issueNumber = element[0]

            for key in folval:
                if isinstance(folval[key], str):
                    continue
                if (issueNumber,) in folval[key]:
                    issuesToRemove.append(issue_to_human_readable(key))
                    break

    for issue in issuesToRemove:
        remove_car_issue(car, issue)

def reset_all():
    i = 1
    while i < NUMBEROFCARS+1:
        car = 'car' + str(i)
        reset_car(car)
        i = i + 1

def which_cars_have(humanReadableIssue):
    issue = human_readable_to_issue(humanReadableIssue)
    carNames = []

    for issueElement in folval[issue]:
        if issueElement == ('',):
            ChatBotPrint("No cars have this issue")
            return
        issueNumber = issueElement[0]
        for beInElement in folval["be_in"]:
            if issueElement == ('',):
                ChatBotPrint("No cars have this issue")
                break
            if issueNumber == beInElement[0]:
                carNames.append(get_car_name_from_abbreviation(beInElement[1]))
    
    if len(carNames) > 0:
        for carName in carNames:
            ChatBotPrint(carName)
    else:
        ChatBotPrint("No cars have this issue")

def list_all_issues():
    i = 1
    while i < NUMBEROFCARS+1:
        car = 'car' + str(i)
        carName = get_car_name(car)
        if carName is None:
            ChatBotPrint(f'{car}: No name set')
        else:
            ChatBotPrint(f'{car}: {carName}')
        
        what_is_wrong_with(car)
        i = i + 1

def get_car_name(car):
    if car in carNameDict:
        return carNameDict[car]

def transformer_predict(sentence):
    ChatBotPrint(transformer.predict(sentence))

def load_q_table():
    if os.path.isfile(QTABLEFILENAME):
        q_table= np.load(QTABLEFILENAME)
    else:
        q_table = np.zeros([env.observation_space.n, env.action_space.n])
    return q_table

def train_taxi_game(times = 10000):

    ChatBotPrint(f'Training for {times} times')

    all_epochs = []
    all_penalties = []

    q_table = load_q_table()

    for i in range(1, int(times)):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = env.step(action) 
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1
            
        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    ChatBotPrint("Training finished.\n")
    np.save(QTABLEFILENAME, q_table)

def print_frames(frames, speed = 0.5):
    os.system("cls")
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        sleep(speed)

def play_the_taxi_game():
    frames = []

    q_table = load_q_table()

    total_epochs, total_penalties = 0, 0

    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    stepCount = 0
    while not done:
        stepCount += 1

        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        frames.append({
            'frame': env.render(mode='ansi'),
            }
        )

        if reward == -10:
            penalties += 1

        epochs += 1

        if (stepCount >= 100):
            print_frames(frames, 0.025)
            ChatBotPrint("Could not complete the game in less than 100 moves")
            ChatBotPrint("Try training the model further")
            return

    print_frames(frames)
    ChatBotPrint(f'Finished in {len(frames)} moves')

def clear_taxi_game_training():
    if os.path.isfile(QTABLEFILENAME):
        os.remove(QTABLEFILENAME)
        ChatBotPrint("Training cleared")

def PostProcessResponse(answer, userInput):
    if answer:
        if IsCommand(answer):
            command = GetCommand(answer)
            values = GetValues(answer)
            if command == ENDSESSIONCMD:
                ChatBotPrint(values[1])
                exit()
            elif command == WIKICMD:
                DisplayWikiResponse(answer, userInput)
            elif command == GOOGLEGETPLACESCMD:
                DisplayGooglePlaces(GetGooglePlacesResponse(values[1]))
            elif command == GOOGLEGETPLACEDETAILSCMD:
                DisplayExtraPlaceDetails(GetGoogleSinglePlaceResponse(ParseGooglePlaceId(GetGooglePlacesResponse(values[1]))))
            elif command == GOOGLETRAVELDISTANCECMD:
                DisplayTravelDistanceToPlace(GetGoogleDistanceAndTravelTime(latlon, ParseGooglePlaceId(GetGooglePlacesResponse(values[1])))) 
            elif command == GOOGLETRAVELTIMECND:
                DisplayTravelTimeToPlace(GetGoogleDistanceAndTravelTime(latlon, ParseGooglePlaceId(GetGooglePlacesResponse(values[1]))))
            elif command == MODELCLASSIFICATIONCMD:
                img = get_image_array(answer)
                if (does_image_exist(img)):
                    ChatBotPrint("The model is " + predictor.predict_car_model(img))
            elif command == CHOOSEMODELIMAGECMD:
                img = get_image_array(ShowImageSelector())
                if (does_image_exist(img)):
                    ChatBotPrint("The model is " + predictor.predict_car_model(img))
            elif command == MANUFACTURERCLASSIFICATIONCMD:
                img = get_image_array(answer)
                if (does_image_exist(img)):
                    ChatBotPrint("The manufacturer is " + predictor.predict_car_manufacturer(img))
            elif command == CHOOSEMANUFACTURERIMAGECMD:
                img = get_image_array(ShowImageSelector())
                if (does_image_exist(img)):
                    ChatBotPrint("The manufacturer is " + predictor.predict_car_manufacturer(img))
            elif command == MODELANDMANUFACTURERCALASSIFICATIONCMD:
                img = get_image_array(answer)
                if (does_image_exist(img)):
                    ChatBotPrint("The manufacturer is " + predictor.predict_car_manufacturer(img))
                    ChatBotPrint("The model is " + predictor.predict_car_model(img))
            elif command == CHOOSEMODELANDMANUFACTURERIMAGECMD:
                img = get_image_array(ShowImageSelector())
                if (does_image_exist(img)):
                    ChatBotPrint("The manufacturer is " + predictor.predict_car_manufacturer(img))
                    ChatBotPrint("The model is " + predictor.predict_car_model(img))
            elif command == SETCARHASISSUE:
                set_car_issue(values[1], values[2])
            elif command == FIXCARISSUE:
                if(remove_car_issue(values[2], values[1])):
                    ChatBotPrint("Issue has been resolved")
                else:
                    ChatBotPrint("Can't remove issue as it was not logged before")
            elif command == DOESCARHAVEISSUE:
                does_car_have_issue(values[1], values[2])
            elif command == WHATISWRONGWITH:
                what_is_wrong_with(values[1])
            elif command == SETCARNAME:
                carNameDict[values[1]] = values[2]
            elif command == GETCARNAME:
                if values[1] in carNameDict:
                    ChatBotPrint(f'{values[1]} is a {carNameDict[values[1]]}')
                else:
                    ChatBotPrint('Car has not been given a name yet')
            elif command == RESETCAR:
                reset_car(values[1])
            elif command == RESETALL:
                reset_all()
            elif command == WHICHCARSHAVE:
                which_cars_have(values[1])
            elif command == LISTALLISSUES:
                list_all_issues()
            elif command == TRAINTAXIGAME:
                train_taxi_game()
            elif command == TRAINTAXIGAMEXTIMES:
                train_taxi_game(values[1])
            elif command == PLAYTAXIGAME:
                play_the_taxi_game()
            elif command == CLEARTAXIGAMETRAINING:
                clear_taxi_game_training()
            elif command == EXPLAINTAXIGAME:
                ChatBotPrint("The taxi game randomly generates an grid of 5 * 5. On this grid is a taxi, represented by the yellow square.")
                ChatBotPrint("There is four other locations on the map marked as R, G, Y, B")
                ChatBotPrint("One of these places will be marked blue when the game starts, this represents a location where a passanger is awaiting pick up")
                ChatBotPrint("Another place will be highlighted purple, this is the passengers desired destination")
                ChatBotPrint("The taxi must navigate to the passengr, pick them up, and drive to the chosen destination, then drop them off")
                ChatBotPrint("The taxi turns into a green square when carrying a passenger")
                ChatBotPrint("The aim of the game is to drop the passenger at the destination in the least amount of moves")
            elif command == UNKNOWNCMD:
                handle_unknown_input(userInput)
        else:
            ChatBotPrint(answer)
    else:
        ChatBotPrint("Please enter something")

def MainLoop():
    while True:
        userInput = GetUserInput()
        answer = kern.respond(userInput)
        PostProcessResponse(answer, userInput)

#######################################################
# Welcome
#######################################################
print("Welcome to the car chat bot. Please feel free to ask questions about",
      "different car parts and where they are located within the car.",
      "I can also identify different models and manufacturers from images!",
      "As well as storing data about which cars need to be fixed, with what issue",
      "You can train me to play the taxi game, ask me to explain it for further details")

#######################################################
#Call Main loop
#######################################################
predictor = Image_Predictor('models\\TransferLearningModelForCarModels.h5', 'models\\SequentialModelForManufacturers.h5')
MainLoop()