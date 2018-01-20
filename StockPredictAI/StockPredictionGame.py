import time

print ("Welcome to the Stock Prediction game")
time.sleep(1) # Delay for 1 second.
input("Please press enter to continue after each statement, including this one.")
print("")
print ("In this game, you will be given $10,000 to be spent on")
print ("a random stock of our choosing. The time at which you decide to buy and sell,")
input ("will be up to you.")
print('')
print ("Each level of the game consists of rounds.")
print ("In each round, you must either Buy, Sell, or Pass (when possible).")
input ("Your goal will be to make as much money as possible, given the information we provide.")
print('')
print ("The Buy option:")
print ("This purchases $1,000 of the stock, at the price currently specified.")
input ("Although you will not be able to buy, should you have less than $1,000 available.")
print('')
print ("The Sell option:")
print ("This sells $1,000 of the stock, at the price currently specified.")
input ("Although you will not be able to sell, should you have less than $1,000 worth of stock available.")
print('')
print ("The Pass option:")
input ("This skips the rounds, and moves to the next one. You may do this as many times as you like.")
print('')
print ("After each round, a graphical plot will show the current price data you have available")
print ("for the specific stock. This may help you decide when to buy.")
input ("You will be given a stock price each week, with the date provided.")
print('')
print ("After all the stock data has been given to you, the level will end.")
print ("At the end of the level, you must have more than $10,000 to continue.")
print ("If you do not, then i'm afraid it's game over for you, and you mast start again.")
print('')
input ("Are you ready?")
print('')
print ("Well ok then let's go!")
time.sleep(1) # Delay for 1 second.
print ("Game starts in 3")
time.sleep(1) # Delay for 1 second.
print ("Game starts in 2")
time.sleep(1) # Delay for 1 second.
print ("Game starts in 1")
time.sleep(1) # Delay for 1 second.

#Intro over
#Starting game level loop

import pandas as pd #import the pandas csv and csv file library
import numpy as np
import matplotlib.pyplot as plt
file = 'stocks_w_NA.csv' #define the imported file with stock data
df = pd.read_csv(file) 
#df = x1.parse("stockData") #create a dataframe of the stockdata sheet
level = 1
coins = 10000
pastPrices = []
pastDates = []
actionChoices = []
stockStartCol = 1
stockEndCol = 5

while level <= 4: #Set how many stocks are available as levels
 subLevel = 0 #initiates the stock round count
 finalSubLevel = 10 #the number of rounds before a new level. Placeholder for eventually equaling totalDates
 print ("Welcome to level: " , level , ")")
 time.sleep(2) # Delay for 2 second.
 print ("We have selected a stock we want you to capitalise on!")
 time.sleep(2) # Delay for 2 second.
 df1 = df.iloc[:,stockStartCol:stockEndCol] #selects a single stock subset from the main csv sheet
 #print (df1)
 totalDates = df1.shape[0] #fetches total number of rows from csv subset
 #print (totalDates)
 stock = 0 #place holder for eventual stock array
 coins = 10000 #need to make sure that at every level we start with 10,000
 while subLevel <= finalSubLevel:
    date = df1.iloc[subLevel,1] #fetches date from the csv subset
    pastDates.append(date)
    print ("Todays date is: " , date , " (year/month/day))")
    price = df1.iloc[subLevel,2]
    pastPrices.append(price)
    plotPrice = price * 2
    time.sleep(2) # Delay for 2 seconds.
    print ("Todays price is: " , price , ")")
    time.sleep(2) # Delay for 2 seconds.
    print ("You have: " , coins , " dollars left)")
    time.sleep(2) # Delay for 2 seconds.
    print ("You have: " , stock , " stock)")
    plt.axis([0, subLevel, 0, plotPrice])
    plt.ion()
    for i in range(subLevel):
        y = pastPrices[i]
        plt.scatter(i , y)
    plt.show()
    while (True):
        print ("Choose either B for buy, S for sell, or P for pass.")
        action = input ("What would you like to do? Press enter to confirm.")
        if (action == 'b'):
            choice = input ("You pressed B. Are you happy with your selection? Press y for yes and n for no.")
            if (choice == 'y'):
                if (coins > 1000):
                    stockPurchase = 1000/price
                    coins -= 1000
                    stock += stockPurchase
                    print ("You have chosen to purchase: %d stock" % (stockPurchase))
                    print ("You have: %d stock" % (stock))
                    print ("You have: %d coins left" % (coins))
                    break
                if (coins < 1000):
                    print ("You do not have enough coins to do that.")
        if (action == 's'):
            choice = input ("You pressed S. Are you happy with your selection? Press y for yes and n for no.")
            if (choice == 'y'):
                stockSell = 1000/price
                if (stock > stockSell):
                    stock -= stockSell
                    coins += 1000
                    print ("You have chosen to sell: %d stock" % (stockSell))
                    print ("You have: %d stock" % (stock))
                    print ("You have: %d coins left" % (coins))
                    break
                if (stock < stockSell):
                    print ("You do not have enough stock to do that.")
        if (action == 'p'):
            choice = input ("You pressed P. Are you happy with your selection? Press y for yes and n for no.")
            if (choice == 'y'):
                break
    if (subLevel == finalSubLevel):
       print ("You have reached the end of this level.")
       print ("All of your remaining stock will now be sold, and your fate will be calculated")
       coins += stock * price
       stock = 0
       if (coins >= 10000):
         print ("You have passed onto the next level! Congratulations.")
       if (coins < 10000):
         print ("You have not managed to pass onto the next level.")
         print ("You must now start from the beggining, and try again.")
         level = 0
         coins = 10000
    subLevel += 1 #increments the a single stock. Equivilant to a round of the game    
 stockStartCol += 4 #increments the first column of the next stock
 stockEndCol += 4 #increments the last column of the next stock    
 level += 1 #increments the levels of the game. Each level is one stock
