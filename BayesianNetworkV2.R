install.packages("BiocManager")
BiocManager::install("Rgraphviz")
install.packages("gRain")
#install.packages("bnlearn")
library(bnlearn)
library(Rgraphviz)
library(gRain)

#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install("Rgraphviz")

#constructing a DAG based on the expert elicitation
#note that the dependencies and the directions of arcs are provided by a domain expert 
dag1 <- empty.graph(nodes = c("A", "S", "E", "O", "R", "T"))
dag1
dag1 <- set.arc(dag1, from = "A", to = "E")
dag1<- set.arc(dag1, from = "S", to = "E")
dag1 <- set.arc(dag1, from = "E", to = "O")
dag1 <- set.arc(dag1, from = "E", to = "R")
dag1 <- set.arc(dag1, from = "O", to = "T")
dag1 <- set.arc(dag1, from = "R", to = "T")
dag1
modelstring(dag1)
nodes(dag1)
arcs(dag1)


#we can determine the states of each node as follows
#note that if you are reading the data from file, you do not need to do this
A.lv <- c("young", "adult", "old")
S.lv <- c("M", "F")
E.lv <- c("high", "uni")
O.lv <- c("emp", "self")
R.lv <- c("small", "big")
T.lv <- c("car", "train", "other")
#the following command plot the DAG structure by highlighting "E" as the most important variable
graphviz.plot(dag1,shape = "ellipse", highlight = list(nodes = "R", col = "blue", fill = "blue"))

#you can manually (custom fit) assign conditional probabities to each node
#however, these probabilities for a more complext and real world problems 
#will be estimated based on some learning algorithms
A.prob <- array(c(0.30, 0.50, 0.20), dim = 3, dimnames = list(A = A.lv))
A.prob
S.prob <- array(c(0.60, 0.40), dim = 2,dimnames = list(S = S.lv))
S.prob
O.prob <- array(c(0.96, 0.04, 0.92, 0.08), dim = c(2, 2), dimnames = list(O = O.lv, E = E.lv))
O.prob
R.prob <- array(c(0.25, 0.75, 0.20, 0.80), dim = c(2, 2), dimnames = list(R = R.lv, E = E.lv))
R.prob
E.prob <- array(c(0.75, 0.25, 0.72, 0.28, 0.88, 0.12, 0.64, 0.36, 0.70, 0.30, 0.90, 0.10), dim = c(2, 3, 2),
                dimnames = list(E = E.lv, A = A.lv, S = S.lv))
T.prob <- array(c(0.48, 0.42, 0.10, 0.56, 0.36, 0.08, 0.58,0.24, 0.18, 0.70, 0.21, 0.09), dim = c(3, 2, 2),
                dimnames = list(T = T.lv, O = O.lv, R = R.lv))
#this command put all the conditional probabilitt tables (cpts) into a single data.frame
cpt <- list(A = A.prob, S = S.prob, E = E.prob, O = O.prob, R = R.prob, T = T.prob)
# we can now add the cpts to the created Dage 
bn <- custom.fit(dag1, cpt)
bn
#the following command will show the marginal probabilities on each node
graphviz.chart(bn, type = "barprob",  grid = TRUE,  bar.col = "darkgreen", strip.bg = "lightskyblue")

graphviz.chart(bn, type = "barprob", bar.col = "gold",strip.bg = "lightskyblue")


nparams(bn)#the total number of parameters
arcs(bn)# the total number of arcs
R.cpt <- coef(bn$R)#this gives you the CPT of Node "R" given its parents (i.e., "E")
R.cpt


#########################################
#learning a Bayesian network (BN) from DATA 
setwd('C:/Users/ac5916/OneDrive - Coventry University/Desktop Jan 2019/Modules/7135CEM/7135CEM_November_2021/Session 4/Lab materials')
survey <- read.table("survey.txt", header = TRUE)#read your data from the working directory
#Since the random variables in Survey data set are Categorical
# we use "factor(variable)" to convert "variable" into a Categorical one 
survey$A<-factor(survey$A)
survey$R<-factor(survey$R)
survey$E<-factor(survey$E)
survey$O<-factor(survey$O)
survey$S<-factor(survey$S)
survey$T<-factor(survey$T)
#survey<-as.factor(survey)
head(survey)
str(survey)# this shows the structure of the data sets

#let assume we know the DAG structure as we have constructed above
# the CPT can be learned using the following command
#note that the parameters (or probabilities) are computed using "MLE"
bn.mle <- bn.fit(dag1, data = survey, method = "mle")
bn.mle
#the following command will show the marginal probabilities on each node
graphviz.chart(bn.mle, type = "barprob",grid = TRUE,  bar.col = "darkgreen",strip.bg = "lightskyblue")

graphviz.chart(bn.mle, type = "barprob",  grid = TRUE,  bar.col = "darkgreen",
               strip.bg = "lightskyblue")

#dag2 =tabu(survey)
#dag2 =hc(survey)
#dag2=hpc(survey)
dag2=hc(survey)
fit2 = bn.fit(dag2, survey, method = "bayes")
fit2

graphviz.chart(fit2, type = "barprob",  grid = TRUE,  bar.col = "darkgreen",
               strip.bg = "lightskyblue")


prop.table(table(survey[, c("O", "E")]), margin = 2)#the contigency table between "O" and "E"
bn.mle$O # the cpt of "Pr(O|E)"

#note that the parameters (or probabilities) are computed using "Bayes" method
#the mean of posterior distribution is considered as the estimate 
# "iss" : a numeric value, the imaginary sample size used by the bayes method to 
#estimate the conditional probability tables associated with discrete nodes
bn.bayes <- bn.fit(dag, data = survey, method = "bayes", iss = 10)
bn.bayes$O

bn.bayes <- bn.fit(dag, data = survey, method = "bayes", iss = 20)
bn.bayes$O
# conditional independence test to check "T" is independent
# from "E" goven "(O, R)"
ci.test("T", "E", c("O", "R"), test = "mi", data = survey)
ci.test("T", "E", c("O", "R"), test = "x2", data = survey)
ci.test("T", "O", "R", test = "x2", data = survey)

arc.strength(dag1, data = survey, criterion = "x2")

score(dag1, data = survey, type = "bic")
score(dag2, data = survey, type = "bic")
score(dag, data = survey, type = "bde", iss = 10)

dag4 <- set.arc(dag, from = "E", to = "T")
nparams(dag4, survey)
score(dag4, data = survey, type = "bic")

rnd <- random.graph(nodes = c("A", "S", "E", "O", "R", "T"))
modelstring(rnd)
score(rnd, data = survey, type = "bic")

learned <- hc(survey) # learning the DAG using hill-climbing (hc) method
modelstring(learned)
score(learned, data = survey, type = "bic")
score(dag, data = survey, type = "bic")



learned2 <- hc(survey, score = "bde")
arc.strength(learned, data = survey, criterion = "bic")
arc.strength(dag, data = survey, criterion = "bic")




bn_model1<- bn.fit(dag, data=survey)
bn_model1
graphviz.chart(bn_model1, type = "barprob",  grid = TRUE,  bar.col = "darkgreen",
               strip.bg = "lightskyblue")
graphviz.chart(bn_model1, type = "barprob", grid = TRUE, bar.col = "gold",
               strip.bg = "lightskyblue")
