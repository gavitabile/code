library(tseries)
library(zoo)
library(quadprog)

SP500Shares <- read.table("C:/Users/gabri/Old Laptop/International Finance/Geisinger/SP500Ticker.csv",header=TRUE,sep=";")
SP500Tickers <- SP500Shares$Symbol

#As a first step of my case study I had to choose around 30 stocks to build my portfolio
#The portfolio's shares choice was wanted to be not random, but with some criteria in the background, so it took kinda long (until row 246 skip if not interested)

SortSectors <- function(SP500Shares) {
  
  NumberOfStocks <- length(SP500Shares$Symbol)
  
  CommunicationServices <- c()
  ConsumerDiscretionary <- c()
  ConsumerStaples <- c()
  Energy <- c()
  Financials <- c()
  HealthCare <- c()
  Industrials <- c()
  InformationTechnology <- c()
  Materials <- c()
  RealEstate <- c()
  Utilities <- c()
  
  for (i in 1:NumberOfStocks) {
    Ticker <- SP500Shares$Symbol[i]
    
    if (SP500Shares$GICS.Sector[i] == "Communication Services") {
      CommunicationServices <- c(CommunicationServices, Ticker)
    } else if (SP500Shares$GICS.Sector[i] == "Consumer Discretionary") {
      ConsumerDiscretionary <- c(ConsumerDiscretionary, Ticker)
    } else if (SP500Shares$GICS.Sector[i] == "Consumer Staples") {
      ConsumerStaples <- c(ConsumerStaples, Ticker)
    } else if (SP500Shares$GICS.Sector[i] == "Energy") {
      Energy <- c(Energy, Ticker)
    } else if (SP500Shares$GICS.Sector[i] == "Financials") {
      Financials <- c(Financials, Ticker)
    } else if (SP500Shares$GICS.Sector[i] == "Health Care") {
      HealthCare <- c(HealthCare, Ticker)
    } else if (SP500Shares$GICS.Sector[i] == "Industrials") {
      Industrials <- c(Industrials, Ticker)
    } else if (SP500Shares$GICS.Sector[i] == "Information Technology") {
      InformationTechnology <- c(InformationTechnology, Ticker)
    } else if (SP500Shares$GICS.Sector[i] == "Materials") {
      Materials <- c(Materials, Ticker)
    } else if (SP500Shares$GICS.Sector[i] == "Real Estate") {
      RealEstate <- c(RealEstate, Ticker)
    } else if (SP500Shares$GICS.Sector[i] == "Utilities") {
      Utilities <- c(Utilities, Ticker)
    }
  }
  
  result <- list(
    CommunicationServices = CommunicationServices,
    ConsumerDiscretionary = ConsumerDiscretionary,
    ConsumerStaples = ConsumerStaples,
    Energy = Energy,
    Financials = Financials,
    HealthCare = HealthCare,
    Industrials = Industrials,
    InformationTechnology = InformationTechnology,
    Materials = Materials,
    RealEstate = RealEstate,
    Utilities = Utilities
  )
  return(result)
}

#The tickers table for S&P500 provided from professor gives indication about the share in the index.
#The information are ticker symbol, name of the company, industry classification, sub-industry and the date the stock was added to S&P.
#I wrote a function in order to classify the stocks in the index according to their sector, 
#using the third coloumn of the table, then i stored them in lists. There is probably a more fast and efficient way to do that 

SharesbySector <- SortSectors(SP500Shares)

CommunicationServices <- SharesbySector$CommunicationServices
ConsumerDiscretionary <- SharesbySector$ConsumerDiscretionary
ConsumerStaples <- SharesbySector$ConsumerStaples
Energy <- SharesbySector$Energy
Financials <- SharesbySector$Financials
HealthCare <- SharesbySector$HealthCare
Industrials <- SharesbySector$Industrials
InformationTechnology <- SharesbySector$InformationTechnology
Materials <- SharesbySector$Materials
RealEstate <- SharesbySector$RealEstate
Utilities <- SharesbySector$Utilities

getPrices <- function(TickerSymbols,start,end,type){
  NumberOfStocks <- length(TickerSymbols)
  prices <- get.hist.quote(TickerSymbols[1],start=start,end=end,quote=type)
  goodSymbols <- TickerSymbols[1]
  for (d in 2:NumberOfStocks) {
    tryCatch({
      P <- get.hist.quote(TickerSymbols[d],start=start,end=end,quote=type)
      prices <- cbind(prices,P) 
      goodSymbols <- c(goodSymbols,TickerSymbols[d])
    }, error=function(err) { print(paste("Download ERROR: ", TickerSymbols[d])) } )
  }
  prices <- data.frame(coredata(prices))
  colnames(prices) <- goodSymbols
  NumberOfGoodStocks <- dim(prices)[2]
  T <- dim(prices)[1]
  badSymbols <- rep(FALSE,NumberOfGoodStocks)
  for (d in 1:NumberOfGoodStocks) {
    if (is.na(prices[1,d]) || is.na(prices[T,d])) {
      badSymbols[d] <- TRUE
    } else {
      if ( sum(is.na(prices[,d]))>0) { 
        print(paste(goodSymbols[d]," NAs filled: ", sum(is.na(prices[,d]))))
        prices[,d]<-na.approx(prices[,d])
      } 
    }
  }
  if (sum(badSymbols)>0){
    prices <- prices[!badSymbols]
    print(paste("Removed due to NAs: ", goodSymbols[badSymbols]))
  }
  if ( sum(is.na(prices))==0 ) {
    if (sum(prices == 0) > 0) {print("Check Zeros!")}
  } else {print("Check NAs and Zeros")}
  prices
}

#ComSerPrices <- getPrices(CommunicationServices,"2019-01-01","2022-12-31","Close")
#match("ATVI",CommunicationServices)
ComSerPrices <- getPrices(CommunicationServices[2:length(CommunicationServices)],"2019-01-01","2022-12-31","Close")
ConDisPrices <- getPrices(ConsumerDiscretionary,"2019-01-01","2022-12-31","Close")
ConStPrices <- getPrices(ConsumerStaples,"2019-01-01","2022-12-31","Close")
EnPrices <- getPrices(Energy,"2019-01-01","2022-12-31","Close")
FinPrices <- getPrices(Financials,"2019-01-01","2022-12-31","Close")
HCPrices <- getPrices(HealthCare,"2019-01-01","2022-12-31","Close")
IndPrices <- getPrices(Industrials,"2019-01-01","2022-12-31","Close")
ITPrices <- getPrices(InformationTechnology,"2019-01-01","2022-12-31","Close")
MatPrices <- getPrices(Materials,"2019-01-01","2022-12-31","Close")
REPrices <- getPrices(RealEstate,"2019-01-01","2022-12-31","Close")
UtPrices <- getPrices(Utilities,"2019-01-01","2022-12-31","Close")

#I used data from 2019 to 2022 to be coherent with the case study, I will choose weights
#according to data until 2022, so  I chose my shares according to data until 2022.

getReturns <- function(prices) {
  NumberOfStocks <- dim(prices)[2];   length <- dim(prices)[1]
  returns <- matrix(rep(0,NumberOfStocks*(length-1)), ncol=NumberOfStocks, nrow=length-1)
  for (ind in 1:NumberOfStocks) {
    returns[,ind] <- diff(log(prices[,ind]))
  }
  returns
}

ComSerRet <- data.frame(getReturns(ComSerPrices))
ConDisRet <- data.frame(getReturns(ConDisPrices))
ConStRet <- data.frame(getReturns(ConStPrices))
EnRet <- data.frame(getReturns(EnPrices))
FinRet <- data.frame(getReturns(FinPrices))
HCRet <- data.frame(getReturns(HCPrices))
IndRet <- data.frame(getReturns(IndPrices))
ITRet <- data.frame(getReturns(ITPrices))
MatRet <- data.frame(getReturns(MatPrices))
RERet <- data.frame(getReturns(REPrices))
UtRet <- data.frame(getReturns(UtPrices))

colnames(ComSerRet) <- colnames(ComSerPrices)
colnames(ConDisRet) <- colnames(ConDisPrices)
colnames(ConStRet) <- colnames(ConStPrices)
colnames(EnRet) <- colnames(EnPrices)
colnames(FinRet) <- colnames(FinPrices)
colnames(HCRet) <- colnames(HCPrices)
colnames(IndRet) <- colnames(IndPrices)
colnames(ITRet) <- colnames(ITPrices)
colnames(MatRet) <- colnames(MatPrices)
colnames(RERet) <- colnames(REPrices)
colnames(UtRet) <- colnames(UtPrices)

#Using getPrices function provided in class I downloaded prices for each sector separately,
#then I calculated returns for each of them through getReturns function, still provided in class

ValueAtRisk <- function(returns, alpha=0.95){
  N <- length(returns)
  sort(coredata(returns))[ceiling((1-alpha)*N)]
}

ExpectedShortfall <- function(returns,alpha=0.95) {
  N <- length(returns)
  sort(coredata(returns))[ceiling((1-alpha)*N)]
  for (i in 1:ceiling((1-alpha)*N)) {
    S <- sum(sort(coredata(returns))[1:ceiling((1-alpha)*N)])
    ES <- S/ceiling((1-alpha)*N)
  }
  ES
}

StM <- function(Returns){

  NumberofStocks <- dim(Returns)[2]
  
  Means <- c()
  Volas <- c()
  SharpeRatio <- c()
  VaR <- c()
  ES <- c()
  
  for (i in 1:NumberofStocks){
  Means <- c(Means,mean(Returns[,i]))
  Volas <- c(Volas,sd(Returns[,i]))
  SharpeRatio <- c(SharpeRatio,Means[i]/Volas[i])
  VaR <- c(VaR,ValueAtRisk(Returns[,i]))
  ES <- c(ES,ExpectedShortfall(Returns[,i]))
  }
  StatMeas <- data.frame(Means,Volas,SharpeRatio,VaR,ES)
  rownames(StatMeas) <- colnames(Returns)
  return(StatMeas)
}

#I used StatMesures function from chapter two exercise to calculate statistical measures
#for all the sectors. I had to modify the function in order to make it work for lists
#of shares and not single shares. 

ComSer_SM <-StM(ComSerRet)
ConDis_SM <-StM(ConDisRet)
ConSt_SM <-StM(ConStRet)
En_SM <-StM(EnRet)
Fin_SM <-StM(FinRet)
HC_SM <-StM(HCRet)
Ind_SM <-StM(IndRet)
IT_SM <-StM(ITRet)
Mat_SM <-StM(MatRet)
RE_SM <-StM(RERet)
Ut_SM <-StM(UtRet)

#The statistical measures where stored in tables so they were easier to compare
#The stocks where chosen "manually", since there are different criteria and it would not
#be easy for me to implement a code to do that. The portfolio is composed of 3 shares
#out of each sector, for a total of 33 stocks. The main idea was to have a diversified portfolio
#composed of shares with a good trade-off between performance and risk. Additionally the diversification
#was preferred to performance, among each sector, the three stocks are chosen from different
#sub-industries even though it means sometimes better performing stocks where excluded from the portfolio.
#I could have chosen different numbers of stocks for each sector weighting them for 
#the weights each sector has in the S&P500 index, but i just chose to proceed in this
#way because it would have took more time and the portfolio would end to have just one
#share for some sectors and I am not sure which of the two options would be more 
#representative of the market. And then for sure there are other ways and different criteria to apply.
#Fundamentals were not considered in selection process.

Portfolio <- c("LNT","GOOG","AWK","AMGN","AJG","AZO","AVY","COP","CL","DHR","EA","LLY","XOM","GRMN","GIS","IEX","LIN","LMT","MCD","MAA","MSI","NDAQ","PG","PGR","PSA","RSG","ROP","SBAC","SHW","SNPS","TMUS","WMB","XEL")

PortfolioPrices <- getPrices(Portfolio,"2019-01-01","2023-12-31","Close")
is.na(PortfolioPrices)
sum(is.na(PortfolioPrices))
PortReturns <- data.frame(getReturns(PortfolioPrices))
colnames(PortReturns) <- colnames(PortfolioPrices)

Port_Prices_23 <- getPrices(Portfolio,"2023-01-01","2023-12-31","Close")

FstPer <- dim(PortfolioPrices)[1]-dim(Port_Prices_23)[1]
SndPer <- FstPer+1

Port_Prices_19_22 <- PortfolioPrices[1:FstPer,]
Port_Prices_23 <- PortfolioPrices[SndPer:dim(PortfolioPrices)[1],]

Port_Ret_19_22 <- data.frame(getReturns(Port_Prices_19_22))
Port_Ret_23 <- data.frame(getReturns(Port_Prices_23))

#data were divided manually to have prices of years 2019 to 2022 and prices of year 2023
#I chose to prices and not returns so there is no issue with the return between the first 
#day of 2023 and the last of 2022, if it should be considered in 2023 returns or not.

PlotCorrelation <- function(cormatrix){
  NumberOfStocks <- dim(cormatrix)[1]
  plot(NumberOfStocks,NumberOfStocks,ylim=c(1,NumberOfStocks),xlim=c(1,NumberOfStocks),cex=0)
  for (i in 1:NumberOfStocks) {
    for (j in 1:NumberOfStocks) {
      if (cormatrix[i,j]>=0.5) color= rgb (1,2*(1-cormatrix[i,j]),0)
      else if (cormatrix[i,j] >= 0) color=rgb (2*cormatrix[i,j],1,0)
      else if (cormatrix[i,j] >= -0.5) color=rgb (0,1,-2*cormatrix[i,j])
      else color=rgb (0,2*(1+cormatrix[i,j]),1)
      points(i,j,col=color,pch=15)
    }
  }
}
PlotCorrelation(cor(Port_Ret_19_22))

#I plotted the correlation matrix of my portfolio to have an idea if portfolio building
#criteria made sense or not. Actually it doesn't look like there's a very low correlation.

colnames(Port_Ret_19_22) <- colnames(Port_Prices_19_22)
colnames(Port_Ret_23) <- colnames(Port_Ret_23)

SP500Prices_19_22 <- getPrices(SP500Tickers,"2019-01-01","2022-12-31","Close")
SP500Returns_19_22 <- coredata(getReturns(SP500Prices_19_22))

Len <- dim(SP500Returns_19_22)[2]

SP_Means_19_22 <- c()
SP_Sigmas_19_22 <- c()

for(i in 1:Len){
  SP_Means_19_22 <- c(SP_Means_19_22,mean(SP500Returns_19_22[,i]))
  SP_Sigmas_19_22 <- c(SP_Sigmas_19_22,sd(SP500Returns_19_22[,i]))
}

NumberOfStocks <- dim(PortReturns)[2]

Means <- c()
Sigmas <- c()

for(i in 1:NumberOfStocks){
  Means <- c(Means,mean(Port_Ret_19_22[,i]))
  Sigmas <- c(Sigmas,sd(Port_Ret_19_22[,i]))
}

plot(SP_Sigmas_19_22,SP_Means_19_22, type="p",pch=1,main="Risk-Return 2019-22",xlab="Volatility",ylab="Mean Returns")
points(Sigmas,Means, col="red",pch=16)

#This plot does not look too good but it is just meant to show where the portfolio stocks 
#are positioned compared to the all S&P500 index on a risk-return graph. None of them has 
#negative return of course, even though there are not spectacular returns is easy to notice 
#how in most of the case stocks with higher returns pay in higher volatility. 
#It would be nice to show this plot for each sector because some stocks look to have
#for the same level of volatility, the same or even higher returns, but that probably
#would've lead to sector or sub-industry overlapping.
#The fact that most of the shares performed in a similar way may explain in part that 
#even pursuing diversification the correlation is still quite high among the stocks.
#Probably looking for good combinations of performances and risks lead to similar behaviors and so higher correlation.

PlotMultiSeries <- function(prices) {
  
  NumberOfStocks <- dim(prices)[2]
  Max <- c()
  Min <- c()
  
  for (i in 1:NumberOfStocks){
    Max <- c(Max, max(prices[,i]/prices[1,i]))
    Min <- c(Min, min(prices[,i]/prices[1,i]))
  }
  Max <- max(Max)
  Min <- min(Min)
  
  plot(prices[,1]/prices[1,1], type="l",ylim=c(0.95*Min,1.05*Max))
  for (i in 2:NumberOfStocks){
    lines(prices[,i]/prices[1,i],col=i)
  }
}

PlotMultiSeries(Port_Prices_19_22)

#This plot is even worse looking, but it is just wanted to explain higher correlation
#values than expected. Not considering the outliers that might have deeper peaks, stocks moved in a very similar way. 

FindWeightsForMu <- function(Mu,CovMat,Means){
  N <- dim(CovMat)[1]
  AMat <- cbind(rep(1,N),Means,diag(1,nrow=N))
  bVec <- c(1,Mu,rep(0,N))
  result <- solve.QP(2*CovMat,rep(0,N),AMat,bVec,2)
  
  weights <- result$solution
  weights <- weights / sum(weights)
  
  result$solution <- weights
  result
}

#I had a little modification to FindWeightsForMu function provided by professor, 
#because the sum of the weights I had back from the function was something like 
#1.000000000000449. So I normalized the weights resulting from the function. Maybe
#it was due to a high and odd number of stocks in the portfolio.

COV <- cov(Port_Ret_19_22)

MUs <- seq(min(Means) + 0.00001, max(Means) - 0.00001, 0.0000005)
SIGs <- c()
Ws <- c()

#I even have increased significantly the number of MUs in order to have more iterations 
#because I only had 103 and I was not happy with that.

for (Mu in MUs){
  Solution <- FindWeightsForMu(Mu,COV,Means)
  SIGs <- c(SIGs,sqrt(Solution$value))
  Ws <- rbind(Ws,Solution$solution)
}

FindWeightsForMinVar <- function(CovMat,Means) {
  N <- dim(CovMat)[1]
  AMat <- cbind(rep(1,N),Means,diag(1,nrow=N))
  bVec <- c(1, 0, rep(0, N))
  result <- solve.QP(2 * CovMat, rep(0, N), AMat, bVec, 1)
  
  weights <- result$solution
  weights <- weights / sum(weights)
  
  result$solution <- weights
  result
}

#FindWeightsForMinVar is basically FindWeightsForMu function but with no specified
#target Mu. FindWeightsForMu gives the weights for the minimum variance possible 
#given a certain Mean Return value. All I did was taking out the target Mu from
#the inputs and gave no specific Mean return as a constraint. Then the function works
#in the same way, solving an optimization problem, but with the only constraints
#that the weights are non-negative and that sum to 1.


MinVar_Weights <- FindWeightsForMinVar(COV,Means)$solution

MinVar <- sqrt(FindWeightsForMinVar(COV,Means)$value)

PortfolioReturns <- function(StockReturns, weights) {
  if (sum(weights)==1 && length(weights)==dim(StockReturns)[2]) {
    NumStocks <- dim(StockReturns)[2]
    Length <- dim(StockReturns)[1]
    P <- rep(0,Length)
    for (t in 1:Length) {
      for (d in 1:NumStocks) {
        P[t] <- P[t] + weights[d]*exp(sum(StockReturns[1:t,d]))	
      }
    }
    P <- c(1,P)
    diff(log(P))
  } else {print("Error: weights do not match")}
}

MinVar_MeanReturn <- mean(PortfolioReturns(Port_Ret_19_22,MinVar_Weights))

SR <- MUs/SIGs
maxSR <- max(SR)
srp <- match(maxSR,SR)

SR_Mu <- MUs[srp]

MaxSR_Weights <- FindWeightsForMu(SR_Mu,COV,Means)$solution

MaxSR_MeanReturn <- mean(PortfolioReturns(Port_Ret_19_22,MaxSR_Weights))

MaxSR_SD <- sqrt(FindWeightsForMu(SR_Mu,COV,Means)$value)

#Portfolio of maximum Sharpe Ratio was found in a raw way. I first used the mean 
#returns and standard deviations of all the portfolios composed of all the possible 
#combinations of weights to calculate the sharpe ratios for all the portfolios.
#Then I extracted the maximum sharpe ratio from the vector. Then I found weights
#and cnsequent standard devition for the matching Mean return. This is the main reason
#I increased the MUs, so that I had more combination and the research of the maximum
#Sharpe Ratio could be a little bit more precise since it was made in a raw way 
#because I was not able to write a function similar to FindWeightsForMu that could
#maximize the Sharpe ratio. The main problem is that i could not use Mean or Sigmas
#vectors but i had to use MUs and SIGs vectors, otherwise I would find the portfolio
#with the maximum weighted average of the assets' sharpe ratios and it would be not
#correct. Those vectors' length was incompatible with variance-covariance matrix dimensions
#and I was not able to go over this problem.

MaxSR_MeanReturn == MUs[srp]
MaxSR_MeanReturn-MUs[srp]

#To be aware of the calculation error I was making I checked the difference between
#the mean return I rawly used to compute the Maximum Sharpe Ratio and the mean return
#of the portfolio composed of the weights I found and they differ for 0.00002010
#so I chose to deal with this difference.

MaxSR_SD == SIGs[srp]

#I made the same for standard deviation and they match.

plot(SIGs,MUs,col="blue",main="Efficient Frontier 2019-22",xlab="Volatility",ylab="Mean Return",xlim=c(0,0.023),ylim=c(0,0.0013))
points(Sigmas,Means, col="black")
points(MinVar,MinVar_MeanReturn,col="red",pch=16)
points(MaxSR_SD,MUs[srp],col="green",pch=16)

#The frontier shows all the portfolios that can be archievable combining
#the 33 assets in different ways. In the graph are highlighted the minimum variance
#portfolio in red and the maximum sharpe ratio portfolio in green. The minimum variance
#portfolio is the portfolio with the minimum global variance possible combining these
#assets, it divides the frontier in two: the superior branch is named Efficient Frontier
#and it includes all the efficient portfolios, while the inferior branch is said to 
#non-efficient because for each level of volatility we find on this branch we can 
#find a different combination of the stocks with the same volatility but higher return
#on the efficient frontier. For the maximum sharpe ratio portfolio point I used MUs[srp]
#value instead of MaxSR_MeanReturn because it was graphically more precise, even though
#it is not exactly the mean return obtained with those specific weights. But as mentioned
#the calculation error is in order of 10^-5.

PlotCorrelation(cor(Port_Ret_23))

#before testing the two portfolios with the actual returns in 2023 I plotted the correlation
#and now diversification looks to be awarded. Previous consideration about risk-return
#and multiseries graphs might have sense looking at the ones of 2023 and comparing to this one.

SP500Prices_23 <- getPrices(SP500Tickers,"2023-01-01","2023-12-31","Close")
SP500Returns_23 <- coredata(getReturns(SP500Prices_23))

LENGTH <- dim(SP500Returns_23)[2]

SP_Means_23 <- c()
SP_Sigmas_23 <- c()

for(i in 1:LENGTH){
  SP_Means_23 <- c(SP_Means_23,mean(SP500Returns_23[,i]))
  SP_Sigmas_23 <- c(SP_Sigmas_23,sd(SP500Returns_23[,i]))
}

Means3 <- c()
Sigmas3 <- c()

for(i in 1:NumberOfStocks){
  Means3 <- c(Means3,mean(Port_Ret_23[,i]))
  Sigmas3 <- c(Sigmas3,sd(Port_Ret_23[,i]))
}

plot(SP_Sigmas_23,SP_Means_23, type="p",pch=1,main="Risk-Return 2023",xlab="Volatility",ylab="Mean Returns")
points(Sigmas3,Means3, col="red",pch=16)

PlotMultiSeries(Port_Prices_23)

#Risk-return plot shows a slight decrease in global volatility and a wider cloud of
#the spots. Talking about Portfolio's share, some of them had negative return in 2023,
#but they still are positioned in the left-middle area of the graph. Already in this
#graph we see the points less close to each other and looking at Multiseries plot
#we can see less overlapping, these behaviors might influence results of correlation.

EW <- c(rep(1/NumberOfStocks,NumberOfStocks))

MinVar_Portfolio <- PortfolioReturns(Port_Ret_23,MinVar_Weights)
MaxSR_Portfolio <- PortfolioReturns(Port_Ret_23,MaxSR_Weights)
EqWeight_Portfolio <- PortfolioReturns(Port_Ret_23,EW)

#Here i calculated the returns for the equally weighted portfolio

Test_Portfolios <- cbind(MinVar_Portfolio,MaxSR_Portfolio,EqWeight_Portfolio)

#the three portfolios were stored in a matrix to be ready to be compared

MinVarP_ToTRet <- exp(sum(MinVar_Portfolio))
MaxSRP_ToTRet <- exp(sum(MaxSR_Portfolio))
EqWeightP_ToTRet <- exp(sum(EqWeight_Portfolio))
ToTRet <- c(MinVarP_ToTRet,MaxSRP_ToTRet,EqWeightP_ToTRet)

t <- dim(Port_Ret_23)[1]

exp(sum(MinVar_Portfolio)) == exp(t*mean(MinVar_Portfolio))
exp(sum(MaxSR_Portfolio)) == exp(t*mean(MaxSR_Portfolio))
exp(sum(EqWeight_Portfolio)) == exp(t*mean(EqWeight_Portfolio))

StatisticalMeasures <- StM(Test_Portfolios)

Comparison <- cbind(ToTRet,StatisticalMeasures)

#In Comparison table we can find for each portfolio Total return and different statistical
#measures. As we could expect Maximum Sharpe Ratio portfolio overperformed the other
#two in total and mean return and in sharpe ratio, the minimum variance portfolio 
#was coherently the one of the three with lowest standard deviation, while for Value
#at Risk and Expected Shortfall the equally weighted portfolio presents lower loss 
#than the the minimum variance portfolio. Even though the differences are very small
#it surprised me a little bit at first, but then I thought it can make sense. In Covid-19
#and Ukrainian war period we had falls of the prices but even recoveries, as we can 
#see in the multiseries plot (this convinced me the most to plot that even if bad 
#looking), while in 2023 shares trend look more smooth. So since volatility consideres
#both positive and negative deviations, while Var and ES only the negative ones it
#makes sense that minimum variance doesn't coincide with the best VaR and ES, and 
#the choice of weights might affect results in 2023.
#To sum up, maximum sharpe ratio portfolio was the one performing the best returns
#and the highest risk among the three, but minimum variance portfolio didn't excel
#in all the risk measures.

NewPortfolioPrices <- getPrices(Portfolio,"2022-01-01","2023-12-31","Close")
is.na(NewPortfolioPrices)
sum(is.na(NewPortfolioPrices))
NewPortfolioReturns <- data.frame(getReturns(NewPortfolioPrices))
colnames(NewPortfolioReturns) <- colnames(NewPortfolioPrices)

NewFstPer <- dim(NewPortfolioPrices)[1]-dim(Port_Prices_23)[1]

Port_Prices_22 <- NewPortfolioPrices[1:NewFstPer,]

Port_Ret_22 <- data.frame(getReturns(Port_Prices_22))

colnames(Port_Ret_22) <- colnames(Port_Prices_22)

PlotCorrelation(cor(Port_Ret_22))

#for the third part of my case study i chose to take year 2022 in order to exclude 
#covid emergency from my previsions and see if the results in 2023 would be the same

SP500Prices_22 <- getPrices(SP500Tickers,"2022-01-01","2022-12-31","Close")
SP500Returns_22 <- coredata(getReturns(SP500Prices_22))

Length <- dim(SP500Returns_22)[2]

SP_Means_22 <- c()
SP_Sigmas_22 <- c()

for(i in 1:Length){
  SP_Means_22 <- c(SP_Means_22,mean(SP500Returns_22[,i]))
  SP_Sigmas_22 <- c(SP_Sigmas_22,sd(SP500Returns_22[,i]))
}

Means2 <- c()
Sigmas2 <- c()

for(i in 1:NumberOfStocks){
  Means2 <- c(Means2,mean(Port_Ret_22[,i]))
  Sigmas2 <- c(Sigmas2,sd(Port_Ret_22[,i]))
}

plot(SP_Sigmas_22,SP_Means_22, type="p",pch=1,main="Risk-Return 2022",xlab="Volatility",ylab="Mean Returns")
points(Sigmas2,Means2, col="red",pch=16)

PlotMultiSeries(Port_Prices_22)

#From risk-return plot global volatility seems to be increased a little bit and mean
#returns seem to be decreased instead, both compared to 2023 or to the 2019-2022 period.
#More of the portfolio assets have negative returns and volatility looks slightly 
#higher than the other graphs too.

NewCOV <- cov(Port_Ret_22)

Mus <- seq(min(Means2) + 0.00001, max(Means2) - 0.00001, 0.0000005)
Sigs <- c()
Weis <- c()

for (Mu in Mus){
  Solution <- FindWeightsForMu(Mu,NewCOV,Means2)
  Sigs <- c(Sigs,sqrt(Solution$value))
  Weis <- rbind(Weis,Solution$solution)
}

MinVar_Ws <- FindWeightsForMinVar(NewCOV,Means2)$solution

MinimumVar <- sqrt(FindWeightsForMinVar(NewCOV,Means2)$value)

MinVar_MeanRet <- mean(PortfolioReturns(Port_Ret_22,MinVar_Ws))

ShRat <- Mus/Sigs
maxShRat <- max(ShRat)
SRP <- match(maxShRat,ShRat)

ShRat_Mu <- Mus[SRP]

MaxSR_Ws <- FindWeightsForMu(ShRat_Mu,NewCOV,Means2)$solution
MaxSR_MeanRet <- mean(PortfolioReturns(Port_Ret_22,MaxSR_Ws))
MaxSR_StDev <- sqrt(FindWeightsForMu(ShRat_Mu,NewCOV,Means2)$value)

MaxSR_MeanRet == Mus[SRP]
MaxSR_MeanRet-Mus[SRP]

MaxSR_StDev == Sigs[SRP]

plot(Sigs,Mus,col="blue",main="Efficient Frontier 2022",xlab="Volatility",ylab="Mean Return")
points(Sigmas2,Means2, col="black")
points(MinimumVar,MinVar_MeanRet,col="red",pch=16)
points(MaxSR_StDev,Mus[SRP],col="green",pch=16)

#shape of the two frontiers are very different. First of all now we even have combinations
#with negative mean return, while before not even the non-efficient frontier had them.
#Then we see the inferior branch much bigger, caused from more stocks with negative
#returns and high volatility

MinVar_Portfolio_New <- PortfolioReturns(Port_Ret_23,MinVar_Ws)
MaxSR_Portfolio_New <- PortfolioReturns(Port_Ret_23,MaxSR_Ws)

Test_Portfolios_New <- cbind(MinVar_Portfolio_New,MaxSR_Portfolio_New,EqWeight_Portfolio)

#Equally weighted portfolio is the same

New_MinVarP_ToTRet <- exp(sum(MinVar_Portfolio_New))
New_MaxSRP_ToTRet <- exp(sum(MaxSR_Portfolio_New))
New_ToTRet <- c(New_MinVarP_ToTRet,New_MaxSRP_ToTRet,EqWeightP_ToTRet)

exp(sum(MinVar_Portfolio_New)) == exp(t*mean(MinVar_Portfolio_New))
exp(sum(MaxSR_Portfolio_New)) == exp(t*mean(MaxSR_Portfolio_New))

StatisticalMeasures2 <- StM(Test_Portfolios_New)

Comparison2 <- cbind(New_ToTRet,StatisticalMeasures2)

#The second comparison surprised me a lot, so much that I still think I made some 
#mistakes in calculation even if I checked 10 times. Equally weighted portfolio over-performed
#the other two portfolios in everything, and performance measures are widely better.
#comparing just the other two portfolios we see maximum SR portfolio performing higher
#returns and sharpe, while minimum variance portfolio performs better in risk measures,
#as might be expected. 
#Probably the stocks that performed worse in 2022 are the ones that performed better 
#in 2023, so based on 2022 previsions I chose smaller weights for those stocks and 
#higher weights for other stocks that performed worse later in 2023. While Equally 
#Weighted portfolio giving equal weights to all of them gained more from the shares
#that then performed good in 2023 and lost less from the shares that then performed
#bad.