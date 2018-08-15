# simple linear regression

# clear console
rm(list=ls())
cat("\014")  

Advertising = read.csv('../data/Advertising.csv', header=TRUE)
#attach(Advertising)

# simple linear regression
lm.fitRadio= lm(sales~radio, data=Advertising)
summary(lm.fitRadio)
lm.fitTV= lm(sales~TV, data=Advertising)
summary(lm.fitTV)

plot(TV, sales)
abline(lm.fitRadio, col='red')
abline(lm.fitTV, col='green')

