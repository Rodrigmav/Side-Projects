# Basic cleaning job #####
library(readr)

#loading in data
Combined_stock_data <- read_csv("~/Documents/Projects for Fun/Proj with Mike/Stock Data/Raw stock data.csv")

#defining other variables I need later
kk <- 1:100000
jj <- 1:100000
data <- Combined_stock_data
data_as_list <- list()
data_long <- data.frame()
data_final <- data.frame()

#Filling in the Stoack name column using the date column
for (jj in 1:24) {
  for (kk in 1:10345)   {
    mm = 4*jj-3
    nn = 4*jj-2
    date <- data[kk, nn]
    if( !is.na(date)) {
      data[kk, mm] = data[1,mm]
    }
  }
}

# Cutting data into pieces and storing it as a list
for (jj in 1:24) {
  mm = 4*jj-3
  nn = 4*jj
  data_as_list[[jj]] <- data[mm:nn]   
}

#assigning names to each stock list
for (jj in 1:24) {
  colnames(data_as_list[[jj]]) <- c("Stock name", "Date", "Low", "Volume")
}

#Making the long list (deleted accidently, remake later)

#changing date format
data_long$Date <- strftime(data_long$Date, '%Y%m%d')

#Putting it back into correct shape
data_final <- data_long[1:10345,]
for (jj in 2:24) {
  mm = 10345*jj-10344
  nn = 10345*jj
  data_final <- cbind(data_final, data_long[mm:nn,])
}
data_final1 <- data_final
#Replace 0 with NA for non blank lines
data_final[data_final== 0] <- NA
  
#saving dataset as csv
write.csv(data_final, 'stocks_w_NA.csv')


# Data mining ######
library(Amelia)

#fixing data types for Amelia
#Low into numeric
data_final[,3] <- as.numeric(data_final[,3])
#Volume is integer class
data_final[,4] = as.numeric(data_final[,4])
data_final[,4] = as.integer(data_final[,4])

#MI with Amelia
a.stock <- amelia(data_final, idvars = c(1:2))



