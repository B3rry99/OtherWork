from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
from datetime import datetime
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import glob
import re
import os
import numpy as np
import sys
nltk.download('punkt')


startTime = datetime.now()

RunCrawler = input('Would you like to run the crawler? (y/n) ')

if RunCrawler == 'y':
    
    AllPub = 'https://pureportal.coventry.ac.uk/en/organisations/school-of-economics-finance-and-accounting/publications/'
    uClient = uReq(AllPub)
    page_html = uClient.read()
    uClient.close()
    page_soup = soup(page_html, "html.parser")
    
    Publication_Title = []
    Author_Names = []
    Date_Published = []
    Publication_Link = []
    Author_Link = []
    Publication_Language = []
    PubID = list(range(638))

    # LOOP FOR EACH PAGE
    # ACQUIRING PUB NAME, PUB LINK AND PUB DATE FOR WHOLE PAGE 
    #  DateContainer = page_soup.findAll("span",{"class":"date"})   # contains dates for all publications on 1st page (50) (WILL BE IN WRONG ORDER)
    PubContainerJ = page_soup.findAll("a",{"rel":"ContributionToJournal"}) # contains link to publication AND publication name (JOUNRALS)
    PubContainerBA = page_soup.findAll("a",{"rel":"ContributionToBookAnthology"}) # contains link to publication AND publication name (BOOK ANTHOLOGY)
    PubContainerC = page_soup.findAll("a",{"rel":"ContributionToConference"}) # contains link to publication AND publication name (CONFERENCE)
    PubContainerWP = page_soup.findAll("a",{"rel":"WorkingPaper"}) # contains link to publication AND publication name (WORKING PAPER)

    # JOIN ALL TYPES OF PAPER
    PubContainer = PubContainerJ + PubContainerBA + PubContainerC + PubContainerWP

    NrOfPubs = page_soup.findAll("div",{"class":"result-container"})
    # LOOP FOR EACH PUB LINK
    # ACQUIRING PUB AUTHORS + DATE
    print("page 0")
    for i in range(len(NrOfPubs)):
        PubLink = PubContainer[i]["href"]  # Change the 0 for each loop (0,1,2,3, etc)
        uClient = uReq(PubLink)
        page_html = uClient.read()
        uClient.close()
        page_soup = soup(page_html, "html.parser")

        # ACQUIRING AUTHORS , PUB DATE , LANGUAGE 
        AuthorContainer = page_soup.findAll("p",{"class":"relations persons"})
        AuthorContainer[0].text # Gives all author's names   
        LangContainer = page_soup.findAll("tr",{"class":"language"}) # Gives original language
        DateContainer = page_soup.findAll("tr",{"class":"status"})
        DateContainer[0].td.text # Gives publication date
        AuthorLink = page_soup.findAll("a",{"class":"link person"})
        if len(AuthorLink) == 2:
            Author1 = AuthorLink[0]["href"]
            AuthorDone = Author1 + "|" + AuthorLink[1]["href"]
        elif len(AuthorLink) == 3:
            Author1 = AuthorLink[0]["href"]
            Author2 = AuthorLink[1]["href"]
            AuthorDone = Author1 + "|" + Author2 + "|" + AuthorLink[2]["href"]
        elif len(AuthorLink) == 1:
            AuthorDone = AuthorLink[0]["href"]        
        else:
            AuthorDone = "No available publisher link"    
        
        Publication_Title.append(PubContainer[i].text)
        Author_Names.append(AuthorContainer[0].text)
        Date_Published.append(DateContainer[0].td.text)
        Publication_Link.append(PubContainer[i]["href"])
        Author_Link.append(AuthorDone)
        Publication_Language.append(LangContainer[0].text.replace("Original language",""))
        print(i,"/49 Documents retrieved" )    

    print("Page 1")

    #############################   ############################# ############################# ############################# ############################# ############################# 

    for g in range(12):
        link1 = 'https://pureportal.coventry.ac.uk/en/organisations/school-of-economics-finance-and-accounting/publications/'
        page = str((g+1))
        LinkDone = 'https://pureportal.coventry.ac.uk/en/organisations/school-of-economics-finance-and-accounting/publications/' + '?page=' + page
        uClient = uReq(LinkDone)
        page_html = uClient.read()
        uClient.close()
        page_soup = soup(page_html, "html.parser")

        PubContainerJ = page_soup.findAll("a",{"rel":"ContributionToJournal"}) # contains link to publication AND publication name (JOUNRALS)
        PubContainerBA = page_soup.findAll("a",{"rel":"ContributionToBookAnthology"}) # contains link to publication AND publication name (BOOK ANTHOLOGY)
        PubContainerC = page_soup.findAll("a",{"rel":"ContributionToConference"}) # contains link to publication AND publication name (CONFERENCE)
        PubContainerWP = page_soup.findAll("a",{"rel":"WorkingPaper"}) # contains link to publication AND publication name (WORKING PAPER)
        PubContainerBA1 = page_soup.findAll("a",{"rel":"BookAnthology"})
        PubContainerP = page_soup.findAll("a",{"rel":"ContributionToPeriodical"})
        PubContainerT = page_soup.findAll("a",{"rel":"Thesis"})
        PubContainerOC = page_soup.findAll("a",{"rel":"OtherContribution"})

        # JOIN ALL TYPES OF PAPER
        PubContainer = PubContainerJ + PubContainerBA + PubContainerC + PubContainerWP + PubContainerBA1 + PubContainerP + PubContainerT + PubContainerOC

        NrOfPubs1 = page_soup.findAll("div",{"class":"result-container"})
        # LOOP FOR EACH PUB LINK
        # ACQUIRING PUB AUTHORS + DATE

        for p in range(len(NrOfPubs1)):
            PubLink = PubContainer[p]["href"]  
            uClient = uReq(PubLink)
            page_html = uClient.read()
            uClient.close()
            page_soup = soup(page_html, "html.parser")

            # ACQUIRING AUTHORS & PUB DATE & LANGUAGE
            AuthorContainer = page_soup.findAll("p",{"class":"relations persons"})
            AuthorContainer[0].text # Gives all author's names

            LangContainer = page_soup.findAll("tr",{"class":"language"}) # Gives original language

            DateContainer = page_soup.findAll("tr",{"class":"status"})
            DateContainer[0].td.text # Gives publication date
            AuthorLink = page_soup.findAll("a",{"class":"link person"})
            if len(AuthorLink) == 2:
                Author1 = AuthorLink[0]["href"]
                AuthorDone = Author1 + "|" + AuthorLink[1]["href"]
            elif len(AuthorLink) == 3:
                Author1 = AuthorLink[0]["href"]
                Author2 = AuthorLink[1]["href"]
                AuthorDone = Author1 + "|" + Author2 + "|" + AuthorLink[2]["href"]
            elif len(AuthorLink) == 1:
                AuthorDone = AuthorLink[0]["href"]        
            else:
                AuthorDone = "No available publisher link"    
            
            Publication_Title.append(PubContainer[p].text)
            Author_Names.append(AuthorContainer[0].text)
            Date_Published.append(DateContainer[0].td.text)
            Publication_Link.append(PubContainer[p]["href"])
            Author_Link.append(AuthorDone)
            Publication_Language.append(LangContainer[0].text.replace("Original language",""))          
            print(p,"/49 Documents retrieved" )
        print("page: ",(g+2))
        #print(datetime.now() - startTime)

    pickle.dump(Publication_Title,open("PubTitles.dat", "wb"))
    pickle.dump(Author_Names,open("AuthorNames.dat", "wb"))
    pickle.dump(Date_Published,open("DatePublished.dat", "wb"))
    pickle.dump(Publication_Link,open("PublicationLinks.dat", "wb"))
    pickle.dump(Author_Link,open("AuthorLinks.dat", "wb"))
    pickle.dump(Publication_Language,open("PublicationLanguage.dat", "wb"))
    #print(datetime.now() - startTime)


########################################### ########################################### ########################################### ###########################################

########################## QUERY SETUP ########################## ##########################

Publication_Title = pickle.load(open("PubTitles.dat", "rb"))
Author_Names = pickle.load(open("AuthorNames.dat", "rb"))
Date_Published = pickle.load(open("DatePublished.dat", "rb"))
Publication_Link = pickle.load(open("PublicationLinks.dat", "rb"))
Author_Link = pickle.load(open("AuthorLinks.dat", "rb"))
Publication_Language = pickle.load(open("PublicationLanguage.dat", "rb"))

Documents = []
for i in range(638):
    proxy = []
    proxy.append(Publication_Title[i])
    proxy.append(Author_Names[i])
    proxy.append(Date_Published[i])
    proxy.append(Publication_Link[i])
    proxy.append(Author_Link[i])
    proxy.append(Publication_Language[i])
    Documents.append(proxy)        
########################## QUERY SETUP ########################## ##########################

###################### REMOVING STOP-WORDS & STEMMING ###################### ##########################
sw = stopwords.words('english')
ps = PorterStemmer()
TitleDONE = []
for doc in Publication_Title:
    tokens = word_tokenize(doc)
    tmp = ""
    for w in tokens:
        if w not in sw:
            tmp += ps.stem(w) + " "
    TitleDONE.append(tmp)
###################### ###################### ######################

########################## INVERTED INDEX ########################## ##########################       
AuthorsNoComma = []
for names in Author_Names:
    AuthorsNoComma.append(names.replace(",",""))
      
inverted_index_title = {}
for i, TitleDONE in enumerate(TitleDONE):
    for term in TitleDONE.split():
        if term in inverted_index_title:
            inverted_index_title[term].add(i)
        else: inverted_index_title[term] = {i}

inverted_index_authors = {}
for i, AuthorsNoComma in enumerate(AuthorsNoComma):
    for term in AuthorsNoComma.split():
        if term in inverted_index_authors:
            inverted_index_authors[term].add(i)
        else: inverted_index_authors[term] = {i}

inverted_index_dates = {}
for i, Date_Published in enumerate(Date_Published):
    for term in Date_Published.split():
        if term in inverted_index_dates:
            inverted_index_dates[term].add(i)
        else: inverted_index_dates[term] = {i}

inverted_index_language = {}
for i, Publication_Language in enumerate(Publication_Language):
    for term in Publication_Language.split():
        if term in inverted_index_language:
            inverted_index_language[term].add(i)
        else: inverted_index_language[term] = {i}

inverted_index = {**inverted_index_title, **inverted_index_authors, **inverted_index_dates, **inverted_index_language}

########################## AND/OR FUNCTIONS ########################## ##########################
def or_postings(posting1, posting2):
    p1 = 0
    p2 = 0
    result = list()
    while p1 < len(posting1) and p2 < len(posting2):
        if posting1[p1] == posting2[p2]:
            result.append(posting1[p1])
            p1 += 1
            p2 += 1
        elif posting1[p1] > posting2[p2]:
            result.append(posting2[p2])
            p2 += 1
        else:
            result.append(posting1[p1])
            p1 += 1
    while p1 < len(posting1):
        result.append(posting1[p1])
        p1 += 1
    while p2 < len(posting2):
        result.append(posting2[p2])
        p2 += 1
    return result

def and_postings(posting1, posting2):
    p1 = 0
    p2 = 0
    result = list()
    while p1 < len(posting1) and p2 < len(posting2):
        if posting1[p1] == posting2[p2]:
            result.append(posting1[p1])
            p1 += 1
            p2 += 1
        elif posting1[p1] > posting2[p2]:
            p2 += 1
        else:
            p1 += 1
    return result
########################## ########################## ########################## 

#################### QUERY FUNCTIONS ########################## ##########################
def and_query(Input):
    TokenInput = word_tokenize(Input)
    while 'AND' in TokenInput: TokenInput.remove('AND')
    resultDone = list()
    finalresult = []
    for i in range((len(TokenInput)-1)):
        p1_1 = list(inverted_index[TokenInput[i]])
        p1_2 = list(inverted_index[TokenInput[i+1]])
        p1_1sorted = sorted(p1_1)
        p1_2sorted = sorted(p1_2)
        result1 = and_postings(p1_1sorted,p1_2sorted)
        resultDone.append(result1)
    flatList = [ item for elem in resultDone for item in elem]    
    for x in flatList:
        if flatList.count(x) == (len(TokenInput)-1):
            finalresult.append(x)    
    finalresults2 = []
    [finalresults2.append(b) for b in finalresult if b not in finalresults2] 
    for docnum in finalresults2:
        a = Documents[docnum]
        print("Publication Title: ")
        print(a[0])
        print("Author(s): ")
        print(a[1])
        print("Date Published: ")
        print(a[2])
        print("Publication Link: ")
        print(a[3])
        print("Author(s) Link: ")
        print(a[4])
        print("Publication Language: ")
        print(a[5])
        print("-----------------------------------------------------------")
    return finalresults2

def or_query(Input):
    TokenInput = word_tokenize(Input)
    while 'OR' in TokenInput: TokenInput.remove('OR')
    for word in TokenInput:
        if word not in inverted_index:
            print('No results found')
            flatList = 0
            return flatList
    resultDone = list()
    finalresult = []
    for i in range((len(TokenInput)-1)):
        p1_1 = list(inverted_index[TokenInput[i]])
        p1_2 = list(inverted_index[TokenInput[i+1]])
        p1_1sorted = sorted(p1_1)
        p1_2sorted = sorted(p1_2)
        result1 = or_postings(p1_1sorted,p1_2sorted)
        resultDone.append(result1)
    flatList = [ item for elem in resultDone for item in elem]      
    finalresults2 = []
    [finalresults2.append(b) for b in flatList if b not in finalresults2] 
    for docnum in finalresults2:
        a = Documents[docnum]
        print("Publication Title: ")
        print(a[0])
        print("Author(s): ")
        print(a[1])
        print("Date Published: ")
        print(a[2])
        print("Publication Link: ")
        print(a[3])
        print("Author(s) Link: ")
        print(a[4])
        print("Publication Language: ")
        print(a[5])
        print("-----------------------------------------------------------")
    return flatList

def single_query(Input):
    if Input not in inverted_index:
        print('No results found')
        finalresults2 = 0
    else:
        finalresults2 = list(inverted_index[Input])
        for docnum in finalresults2:
            a = Documents[docnum]
            print("Publication Title: ")
            print(a[0])
            print("Author(s): ")
            print(a[1])
            print("Date Published: ")
            print(a[2])
            print("Publication Link: ")
            print(a[3])
            print("Author(s) Link: ")
            print(a[4])
            print("Publication Language: ")
            print(a[5])
            print("-----------------------------------------------------------")
    return finalresults2


########################## ########################## ##########################


########################## USER INTERFACE ########################## ##########################



cont = "y"
while cont == "y":
    UserInput = input('Please enter your query: ')
    if "AND" in UserInput:
        and_query(UserInput)
        cont = input('Would you like to search another query? (y/n) ')
    elif "OR" in UserInput:
        or_query(UserInput)
        cont = input('Would you like to search another query? (y/n) ')
    else:
        single_query(UserInput)
        cont = input('Would you like to search another query? (y/n) ')
    
########################## ########################## ##########################


























