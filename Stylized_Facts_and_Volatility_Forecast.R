library(ggplot2)
library(quantmod)
library(rugarch)
library(moments)





getSymbols("JNJ", src = "yahoo")
JNJ <- Cl(JNJ)
JNJ_Returns <- periodReturn(JNJ, period = "daily", type = "log", leading = TRUE)
plot(JNJ,main="Johnson & Johnson Price",xlab="Date",ylab="Price",lwd =2)

plot(JNJ_Returns,main="Johnson & Johnson Returns",xlab="Date",ylab="Returns",ylim=c(-0.12,0.13),lwd =1)

chart_Series(JNJ)
add_TA(JNJ_Returns)

acf(JNJ)
acf(JNJ_Returns)
acf(JNJ_Returns,main="Johnson & Johnson Returns Autocorrelation", lag = 250)
acf(abs(JNJ_Returns),main="Johnson & Johnson Absolute Returns Autocorrelation", lag = 250)

JNJ_mu = mean(JNJ_Returns)
JNJ_sigma = sd(JNJ_Returns)

x_JNJ = seq(-0.05, 0.05, 0.001)
Gaussian_density_JNJ = dnorm(x_JNJ, JNJ_mu, JNJ_sigma)
plot(Gaussian_density_JNJ, type = "l") 

kernel_density_JNJ = density(JNJ_Returns)
plot(kernel_density_JNJ,main="Johnson & Johnson Returns Distribution",xlab="Returns",col="blue",lwd = 2,xlim = range(JNJ_Returns))  
lines(x_JNJ, Gaussian_density_JNJ, col="red",lwd=2,lty = 2)  
legend("topright", legend = c("Johnson & Johnson", "Normal"), col = c("blue", "red"), lwd = 2,lty = c(1, 2))

kurtosis(JNJ_Returns)
skewness(JNJ_Returns)


chart_Series(JNJ_Returns)

rw_size = 120  
Sample_Volatility_JNJ = JNJ_Returns*0
for (i in rw_size:length(JNJ_Returns)){
  Sample_Returns_JNJ = JNJ_Returns[(i - rw_size+1):i]
  Sample_Volatility_JNJ[i] = sd(Sample_Returns_JNJ)* sqrt(250)
}
add_TA(Sample_Volatility_JNJ)

VaR_Forecast_JNJ = JNJ_Returns*0
for (i in rw_size:(length(JNJ_Returns)-1)){
  Sample_Returns_JNJ = JNJ_Returns[(i - rw_size+1):i]
  JNJ_mu = mean(Sample_Returns_JNJ)
  JNJ_sigma = sd(Sample_Returns_JNJ)
  VaR_Forecast_JNJ[i+1] = JNJ_mu - 1.645*JNJ_sigma
}
add_TA(VaR_Forecast_JNJ, on = 1)

JNJ_violations = 0
for (i in rw_size:(length(JNJ_Returns)-1)){
  JNJ_violations = JNJ_violations + (as.numeric(VaR_Forecast_JNJ[i+1]) > as.numeric(JNJ_Returns[i]))
}
JNJ_totals = (length(JNJ_Returns)-1) - rw_size +1
JNJ_violations / JNJ_totals


VaR_GARCH_Forecast_JNJ = JNJ_Returns*0
for (i in rw_size:(length(JNJ_Returns)-1)){
  Sample_Returns_JNJ = JNJ_Returns[(i - rw_size+1):i]
  garch.setup = ugarchspec(mean.model = list(armaOrder = c(0,0), include.mean = TRUE),
                           variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                           distribution.model = "norm") 
  tryCatch({fit_JNJ = ugarchfit(garch.setup, data = Sample_Returns_JNJ, solver = "hybrid")},
           error = function(e) e, warning = function(w) w)
  sim_returns_JNJ = fitted(ugarchsim(fit_JNJ, n.sim = 1, n.start = 0, m.sim = 1000, startMethod = "sample"))

  mu_JNJ = mean(sim_returns_JNJ) 
  sigma_JNJ = sd(sim_returns_JNJ) 
  VaR_GARCH_Forecast_JNJ[i+1] = mu_JNJ - 1.645*sigma_JNJ 
  
  print(i)
}
add_TA(VaR_GARCH_Forecast_JNJ, on=1)

JNJ_GARCH_violations = 0
for (i in rw_size:(length(JNJ_Returns)-1)){
  JNJ_GARCH_violations = JNJ_GARCH_violations + (as.numeric(VaR_GARCH_Forecast_JNJ[i+1]) > as.numeric(JNJ_Returns[i]))
}
JNJ_totals = (length(JNJ_Returns)-1) - rw_size +1
JNJ_GARCH_violations / JNJ_totals


plot(VaR_Forecast_JNJ > JNJ_Returns)
plot(VaR_GARCH_Forecast_JNJ > JNJ_Returns)







getSymbols("BAC", src = "yahoo")
BAC <- Cl(BAC)
BAC_Returns <- periodReturn(BAC, period = "daily", type = "log", leading = TRUE)
plot(BAC,main="Bank of America Corporation Price",xlab="Date",ylab="Price",lwd =2)
plot(BAC_Returns,main="Bank of America Corporation Returns",xlab="Date",ylab="Returns",ylim=c(-0.37,0.33),lwd =1)

chart_Series(BAC)
add_TA(BAC_Returns)

acf(BAC)
acf(BAC_Returns)
acf(BAC_Returns,main="Bank of America Returns Autocorrelation", lag = 250)
acf(abs(BAC_Returns),main="Bank of America Absolute Returns Autocorrelation", lag = 250)

BAC_mu = mean(BAC_Returns)
BAC_sigma = sd(BAC_Returns)

x_BAC = seq(-0.15, 0.15, 0.001)
Gaussian_density_BAC = dnorm(x_BAC, BAC_mu, BAC_sigma)
plot(Gaussian_density_BAC, type = "l")

kernel_density_BAC = density(BAC_Returns)
plot(kernel_density_BAC,main="Bank of America Returns Distribution",xlab="Returns",col="blue",lwd = 2,xlim = range(BAC_Returns))  
lines(x_BAC, Gaussian_density_BAC, col="red",lwd=2,lty = 2)  
legend("topright", legend = c("Bank of America", "Normal"), col = c("blue", "red"), lwd = 2,lty = c(1, 2))

kurtosis(BAC_Returns)
skewness(BAC_Returns)



chart_Series(BAC_Returns)

rw_size = 120  
Sample_Volatility_BAC = BAC_Returns*0
for (i in rw_size:length(BAC_Returns)){
  Sample_Returns_BAC = BAC_Returns[(i - rw_size+1):i]
  Sample_Volatility_BAC[i] = sd(Sample_Returns_BAC)* sqrt(250)
}
add_TA(Sample_Volatility_BAC)

VaR_Forecast_BAC = BAC_Returns*0
for (i in rw_size:(length(BAC_Returns)-1)){
  Sample_Returns_BAC = BAC_Returns[(i - rw_size+1):i]
  BAC_mu = mean(Sample_Returns_BAC)
  BAC_sigma = sd(Sample_Returns_BAC)
  VaR_Forecast_BAC[i+1] = BAC_mu - 1.645*BAC_sigma
}
add_TA(VaR_Forecast_BAC, on = 1)

BAC_violations = 0
for (i in rw_size:(length(BAC_Returns)-1)){
  BAC_violations = BAC_violations + (as.numeric(VaR_Forecast_BAC[i+1]) > as.numeric(BAC_Returns[i]))
}
BAC_totals = (length(BAC_Returns)-1) - rw_size +1
BAC_violations / BAC_totals

plot(VaR_Forecast_BAC > BAC_Returns)

VaR_GARCH_Forecast_BAC = BAC_Returns*0
for (i in rw_size:(length(BAC_Returns)-1)){
  Sample_Returns_BAC = BAC_Returns[(i - rw_size+1):i]
  garch.setup = ugarchspec(mean.model = list(armaOrder = c(0,0), include.mean = TRUE),
                           variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                           distribution.model = "norm")
  tryCatch({fit_BAC = ugarchfit(garch.setup, data = Sample_Returns_BAC, solver = "hybrid")},
           error = function(e) e, warning = function(w) w)
  sim_returns_BAC = fitted(ugarchsim(fit_BAC, n.sim = 1, n.start = 0, m.sim = 1000, startMethod = "sample"))
  
  mu_BAC = mean(sim_returns_BAC) 
  sigma_BAC = sd(sim_returns_BAC) 
  VaR_GARCH_Forecast_BAC[i+1] = mu_BAC - 1.645*sigma_BAC 
  
  print(i)
}
add_TA(VaR_GARCH_Forecast_BAC, on=1)

BAC_GARCH_violations = 0
for (i in rw_size:(length(BAC_Returns)-1)){
  BAC_GARCH_violations = BAC_GARCH_violations + (as.numeric(VaR_GARCH_Forecast_BAC[i+1]) > as.numeric(BAC_Returns[i]))
}
BAC_totals = (length(BAC_Returns)-1) - rw_size +1
BAC_GARCH_violations / BAC_totals

plot(VaR_Forecast_BAC > BAC_Returns)
plot(VaR_GARCH_Forecast_BAC > BAC_Returns)




cor(JNJ_Returns,BAC_Returns)
