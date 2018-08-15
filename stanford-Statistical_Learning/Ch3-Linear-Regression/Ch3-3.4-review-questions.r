# clear console
rm(list=ls())
cat("\014") 

Credit = read.csv('../data/Credit.csv', header=TRUE)

# According to the balance vs ethnicity model, what
# is the predicted balance for an Asian in the data set? (within 0.01 accuracy)
# answer is 512.31 (look on the summary)

# What is the predicted balance for an African American? (within .01 accuracy)
# answer is 531.00 (look on the summary, it's a baseline)

lm.fit = lm(Balance ~ Ethnicity, data = Credit)
summary(lm.fit)
