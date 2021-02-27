import random
import pickle
from os import listdir
from os.path import isfile, join
import re
from nltk import tokenize

def LoadModels():
    #load all 1vsAll models
    models = []
    for categoryindex in range(0,10):
        filename = 'models/Label'+str(categoryindex)+'_new_model.pkl'
        with open(filename, 'rb') as file:
            models.append( pickle.load(file) )
    print(len(models), 'models loaded')
    return models


def SentenceContainsWords(sentence, lW):
    for w in lW:
        w = w.lower().strip()
        if(w in sentence):
            return True

    return False


def SentenceContainsWordsNice(sentence, lW):
    sentence_unigrams  = sentence.split()
    sentence_bigrams = [' '.join(b) for l in sentence for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    #print(sentence)
    for w in lW:
        w = w.lower().strip()
        sizew = len(w.split(' '))
        comparewith = sentence_unigrams
        if(sizew == 1):
            comparewith = sentence_unigrams
        elif(sizew == 2):
            comparewith = sentence_bigrams
        else:
            comparewith = sentence

        if(w in comparewith):
            return True

    return False

#indextolabel
indexToLabel = {0: 'Amazon Pay', 1: 'Device Address', 2: 'Device country and postal code',
                3: 'Email Address', 4: 'Location Services', 5: 'Mobile Number', 6: 'Name',
                7: 'Personal Information', 8: 'Skill Personisation', 9:'None'}

#maybe remove not,
negations = ["No","Not","None","No one","Nobody","Nothing","Neither",
             "Nowhere","Never","Doesn't","Isn't","Wasn't","Shouldn't",
             "Wouldn't","Couldn't","Won't","Can't","Don't",
             "Does not", "Is not", "Was not", 'do not']

keywords = ['location-service', 'location-service','geo', 'location','demographic','gps','wifi', 'wi-fi', 'bluetooth',
'pay','payment', 'amazon-pay', 'amazon_pay', 'credit', 'billing', 'billing information',
'phone', 'number', 'telephone',
'email', 'email_address', 'email-address','e-mail','emails',
'first_name', 'first-name', 'name','given-name', 'given_name','names',
'address', 'home', 'home_address', 'device_address','device-address','addresses',
'contact',
'contact_zip', 'contact-zip', 'city','zip', 'postcode', 'postal_code', 'postal-code','postal', 'street',
'personal','pii','country',
'skill_personilisation', 'skill_personization', 'prediction', 'personilisation', 'personalization', 'personalised', 'personalized']


#ip might give some false negatives, but
#if a skill is collecting an specific privacy policy its very rare,
keywords_ignore = ['call us', 'call me', 'you can contact', 'contact me', 'reach me', 'reach us', 'email us', 'send us',
                   'send me', 'sending us', 'sending an email', 'sending us an email', 'reach us', 'email me','write to us','email at',
                   'problem', 'problems', 'issue', 'issues', 'protocol', 'internet', 'ip address','ip addresses','optout','feedback','contact from',
                   'fax', 'concern', 'unsubscribe', 'opt out', 'below', 'our emails', 'sessions', 'session','contacting us','inquiries','corporate'
                   'issues', 'technical', 'inquiries', 'please contact', 'protocol address', 'please go', 'internet protocol','contact form',
                   'web address', 'address bar', 'headquarters', 'website address', 'mobile app', 'given above','copyright','mail us',
                   'following address', 'writing to us', 'send us', 'post to', 'address of the site', 'personal care', 'phone us',
                   'comment','portfolio', 'domain name','via telephone at', 'connect via','contact uѕ',]


models = []

def AnalyseText(textL, models, smartNegation=False, smartNone=False, kwfilter = False, kw_ignore_filter=False):
    '''
    smartNegation : bool - If True then remove all tags except 'None' if a negation is found in the sentence parsed.
    E.g. "We don't request for credit card information and payment" is tagged as 'Amazon Pay'. However, since there is a negation,
        'Amazon Pay' will be removed and replaced by 'None' if not other label was used to tag the sentence

    If smartNone : bool - If True means that if a sentence is tagged with class None, all other classes tagging the sentence are
    ignored. This is supported by the almost perfect f1 score and accuraccy obtained by class None classifier while the other classes
    struggle a little with None sentences.

    kwfilter : bool - Apply a keyword filtering on the sentences before applying the ML models. Sentences that do not include any of
    the keywords are automatically tagged as class None. This is good to focus the ML classifiers only among these sentences that
    can potentially be interesting. It's a way to remove false positives, consequence of a small training dataset.

    kw_ignore_filter : bool - If True, apply a kw filtering. If a sentence contains any of the elements of the kw_ignore_filter list,
    the sentence is automatically tagged as None. This is good to filter out sentences such as 'contact us' or other sentences often
    found in certain policies that are often wrongly tagged by the classifier.
    '''
    countOverrideNone = 0
    countSentencesProcessedTotal = 0
    countSentencesTotal = 0

    allpermreq = set([])
    line_req = []
    countSentencesTotal+= len(textL)
    for line in textL:
        allpermline = set([])
        line = line.strip()

        proceed = True
        if(kwfilter is True):
            proceed = False
            line = line.lower()
            proceed = SentenceContainsWords(line, keywords)

        if(kw_ignore_filter is True and proceed is True):
            proceed = False
            line = line.lower()
            proceed = not SentenceContainsWords(line, keywords_ignore)

        if(proceed):
            for labelindex, model in enumerate(models):
                labelpredicted = model.predict([line])
                if(labelpredicted == 0):
                    allpermline.add( indexToLabel[labelindex] )

            if(smartNegation is True):
                line = line.lower()
                for n in negations:
                    n = n.lower()
                    if(n in line):
                        #print('\t Applying reverse negation to \"', line, '"')
                        allpermline = set(['None'])

            if(smartNone is True):
                if('None' in allpermline):
                    if(len(allpermline)>1):
                        #None override decision
                        countOverrideNone+=1
                    allpermline = set(['None'])

            countSentencesProcessedTotal+=1



        if(len(allpermline)==0):
            allpermline = set(['None'])
        #print('\t', allpermline, line)
        line_req.append( str(allpermline) +":"+ str(line) )
        allpermreq.update(allpermline)

    return [allpermreq, line_req, (countOverrideNone, countSentencesProcessedTotal, countSentencesTotal)]

def PreprocessIntoLines(text):
    #text = tokenize.sent_tokenize(text[0])
    text = re.split('\||\.|\s{2,}|\>|\<|\?|\!|;|\n |:',text[0])
    return text

def AnalysePolicy(policypath, models, smartNegation=False, smartNone=False, kwfilter = False, kw_ignore_filter=False, splitintolines = False):
    try:
        with open(policypath, encoding='utf8') as f:   #utf8
            text = f.readlines()
            if(splitintolines):
                if(len(text)==1):
                    text = PreprocessIntoLines(text)
                #print(text)

            allpermreq = AnalyseText(text,
                                     models,
                                     smartNegation=smartNegation,
                                     smartNone=smartNone,
                                     kwfilter=kwfilter,
                                     kw_ignore_filter=kw_ignore_filter)

        return allpermreq

    except Exception as e:
        print('ERROR',e)
    return None
