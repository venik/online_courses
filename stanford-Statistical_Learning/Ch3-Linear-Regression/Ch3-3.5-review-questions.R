rm(list=ls())
cat("\014") 

Advertising = read.csv('../data/Advertising.csv', header=TRUE)

lm.fit = lm(sales ~ TV*radio, data=Advertising)
summary(lm.fit)
# According to the model for sales vs TV interacted with radio, what is the effect
# of an additional $1 of radio advertising if TV=$50? (with 4 decimal accuracy)
#lm.fit = lm(sales ~ TV + radio + TV*radio, data=Advertising)
tv50radio0 <- data.frame(TV=50, radio=0)
tv50radio1 <- data.frame(TV=50, radio=1)
res50 <- predict(lm.fit, tv50radio1) - predict(lm.fit, tv50radio0)
round(res50, 4)

# What if TV=$250? (with 4 decimal accuracy)
tv250radio0 <- data.frame(TV=250, radio=0)
tv250radio1 <- data.frame(TV=250, radio=1)
res250 <- predict(lm.fit, tv250radio1) - predict(lm.fit, tv250radio0)
round(res250, 4)
