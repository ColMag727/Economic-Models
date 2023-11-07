""" Magie Zheng xxz896
Weekly project 2. The Gale-Shapley algorithm
Due on: September 13, 11:59 PM
"""
import json
# Section 1. Preparation
# 1-1. import all the necessary python modules


# 1-2. import the datasets
with open('project2_data-1.json') as f:
    preference = json.load(f)
    # Section 2 Extract information from the dataset, 
    # 2-1. create a dictionary 'guyprefer' contains mens' preferences
    guyPrefers = preference['men_preference']
    
    # 2-2. create a dictionary 'galprefer' contains women's preferences
    galPrefers = preference['women_preference']
    # 2-3. create a list contains guys who are currently not engaged, 
    # sort alphabetically
    
    free_guy = galPrefers['Cara'].copy()
    
    
    free_guy.sort()
    

    # 2-4. generate an empty dictionary 'engage_book' to store result
    engage_book = {}
    
    for key in galPrefers:
        
        engage_book[key] = None
        
    # 2-5. make copies of guyprefers and gal refers
    guyPreference = guyPrefers.copy()
    galPreference = galPrefers.copy()
    
    # Section 3. Impletement the Gale-Shapley algorithm 
    # Follow the algorithm flowchart, it should be very helpful
    
    while len(free_guy) != 0:
    # pop the first guy in the free_guy list, let him take the move
    
        a_brave_guy = free_guy.pop(0)
        
        # get his preference list
        
        myList = guyPreference[a_brave_guy].copy()
        
        # print(len(free_guy))
        
        # print("M "+ a_brave_guy)
        
        
        # let this guy take the move
        while  myList:
            # Let's propose to my favorate lady!
            
            my_girl = myList.pop(0)
            
            thisGalPref = galPreference[my_girl]
            
            # print("G "+ my_girl)
            
            # print(engage_book)
            
            # print(thisGalPref)
            
            if engage_book[my_girl] is None:
                
                engage_book[my_girl] = a_brave_guy
                
                # print(my_girl+" & "+ engage_book[my_girl])
                
                break
                
            elif thisGalPref.index(engage_book[my_girl]) > thisGalPref.index(a_brave_guy):
                
                free_guy.append(engage_book[my_girl])
                
                engage_book[my_girl] = a_brave_guy
                
                # print(my_girl+" $ "+ engage_book[my_girl])
                
                break

    
    # student_list.remove[0]
    # # Section 4 (optional). Stability check
    # # define stability: there are no two people of opposite sex who would both
    # # rather have each other than their current partners.

sortBook = sorted(engage_book).copy()

# engageList = list(engage_book.values()).copy()

# boyEngagingInd = engage_book.keys()[engage_book.values().index(16)]

# print(boyEngagingInd)

for i in range(len(engage_book)):
    
    this_girl = sortBook[i]
    
    thisGlPref = galPreference[this_girl]
    
    boyIndex = thisGlPref.index(engage_book[this_girl])
    
    temp = boyIndex - 1
    
    while temp >= 0:
        
        thisBoy = thisGlPref[temp]
        
        thisBoyPref = guyPreference[thisBoy]
        
        this_girl_Ind = thisBoyPref.index(this_girl)
        
        boyEngagingInd = None
        
        for y in range(len(sortBook)):
            
            if engage_book[sortBook[y]] == thisBoy:
                
                boyEngagingInd = thisBoyPref.index(sortBook[y])
                
                break
            
        if boyEngagingInd > this_girl_Ind:
            
            print('there is something wrong')
            
        elif i == 29 and temp ==0:
            
            print('it seems good')
        
        temp -= 1
        








