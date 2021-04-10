path <- '../experiments/2021-04-10T14-20/training.csv'
df <- read.csv(path)

library('ggplot2')
library('dplyr')

ggplot(df, aes(x=epoch)) + 
  #geom_point(aes(y=d_acc_fake), color='red', size=.5, alpha=.5) + 
  geom_smooth(aes(y=d_acc_fake), se=FALSE, color='red') +
  #geom_point(aes(y=g_acc), color='blue', size=.5, alpha=.5) +
  geom_smooth(aes(y=g_acc), se=FALSE, color='blue')
